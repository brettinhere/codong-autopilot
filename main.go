package main

import (
	"context"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha1"
	"database/sql"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math/big"
	"net/http"
	"net/url"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/joho/godotenv"
	"github.com/redis/go-redis/v9"
	_ "modernc.org/sqlite"
)

// ─── globals ────────────────────────────────────────────────────────────────

var (
	db  *sql.DB
	rdb *redis.Client
	ctx = context.Background()
	mu  sync.Mutex // SQLite single-writer

	logCh    = make(chan string, 256)
	eventCh  = make(chan Event, 256)
	wsMu     sync.Mutex
	wsConns  []chan string

	// 各引擎独立开关，默认全部关闭
	enginePublish bool
	engineReply   bool
	engineComment bool
	engineMu      sync.Mutex
)

type Event struct {
	Ev   string      `json:"event"`
	Data interface{} `json:"data"`
	Time string      `json:"time"`
}

// ─── main ────────────────────────────────────────────────────────────────────

func main() {
	godotenv.Load("./autopilot.env")

	initDB()
	initRedis()

	go logWorker()
	go publishEngine()
	go replyEngine()
	go commentEngine()

	port := getenv("DASHBOARD_PORT", "8083")
	logMsg("🤖 Codong Autopilot v1.1 启动 → http://localhost:" + port)
	logMsg("   平台: " + getenv("TARGET_PLATFORMS", "twitter"))
	logMsg("   ⏸️  引擎已暂停，请在控制台点击「启动引擎」开始工作")

	http.HandleFunc("/", serveHTML)
	http.HandleFunc("/ws", serveWS)
	http.HandleFunc("/api/queue/add", handleQueueAdd)
	http.HandleFunc("/api/queue", handleQueueList)
	http.HandleFunc("/api/snapshot", handleSnapshot)
	http.HandleFunc("/api/config", handleConfig)
	http.HandleFunc("/api/config/save", handleConfigSave)
	http.HandleFunc("/api/logs", handleLogs)
	http.HandleFunc("/api/engine/start", handleEngineStart)
	http.HandleFunc("/api/engine/stop", handleEngineStop)
	http.HandleFunc("/api/engine/status", handleEngineStatus)

	log.Fatal(http.ListenAndServe(":"+port, nil))
}

// ─── DB ─────────────────────────────────────────────────────────────────────

func initDB() {
	var err error
	db, err = sql.Open("sqlite", "file:autopilot.db?cache=shared&_journal_mode=WAL")
	must(err)
	db.SetMaxOpenConns(1)

	migrations := []string{
		`CREATE TABLE IF NOT EXISTS post_queue (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			platform TEXT NOT NULL, content TEXT NOT NULL,
			publish_at TEXT NOT NULL, status TEXT DEFAULT 'ready',
			post_id TEXT, error_msg TEXT, retry_count INTEGER DEFAULT 0,
			created_at TEXT NOT NULL, published_at TEXT)`,
		`CREATE TABLE IF NOT EXISTS comment_log (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			platform TEXT NOT NULL, target_post_id TEXT, keyword TEXT,
			comment_text TEXT NOT NULL, status TEXT DEFAULT 'pending',
			posted_at TEXT, error_msg TEXT, created_at TEXT NOT NULL)`,
		`CREATE TABLE IF NOT EXISTS reply_log (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			platform TEXT NOT NULL, post_id TEXT, comment_id TEXT,
			comment_text TEXT, reply_text TEXT, status TEXT DEFAULT 'pending',
			replied_at TEXT, created_at TEXT NOT NULL)`,
		`CREATE TABLE IF NOT EXISTS platform_stats (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			platform TEXT NOT NULL, followers INTEGER DEFAULT 0,
			likes INTEGER DEFAULT 0, comments INTEGER DEFAULT 0,
			views INTEGER DEFAULT 0, recorded_at TEXT NOT NULL)`,
	}
	for _, m := range migrations {
		db.Exec(m)
	}
}

func initRedis() {
	rdb = redis.NewClient(&redis.Options{Addr: "localhost:6379"})
	if err := rdb.Ping(ctx).Err(); err != nil {
		log.Printf("⚠️  Redis 连接失败: %v", err)
	}
}

// ─── utils ───────────────────────────────────────────────────────────────────

func getenv(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func must(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

func logMsg(msg string) {
	line := time.Now().Format("15:04:05") + "  " + msg
	log.Println(msg)
	select {
	case logCh <- line:
	default:
	}
	broadcastEvent(Event{Ev: "log", Data: map[string]string{"message": msg}, Time: time.Now().Format("15:04:05")})
}

func broadcastEvent(e Event) {
	b, _ := json.Marshal(e)
	s := string(b)
	wsMu.Lock()
	defer wsMu.Unlock()
	for _, ch := range wsConns {
		select {
		case ch <- s:
		default:
		}
	}
}

func randomSleep(minSec, maxSec int) int {
	n, _ := rand.Int(rand.Reader, big.NewInt(int64(maxSec-minSec)))
	wait := minSec + int(n.Int64())
	time.Sleep(time.Duration(wait) * time.Second)
	return wait
}

func todayPrefix() string { return time.Now().Format("2006-01-02") }

// ─── LLM (Claude) ────────────────────────────────────────────────────────────

type claudeResp struct {
	Content []struct {
		Text string `json:"text"`
	} `json:"content"`
}

func llmAsk(prompt string) string {
	dailyMax, _ := strconv.Atoi(getenv("LLM_DAILY_LIMIT", "200"))
	today := todayPrefix()
	cntStr, _ := rdb.Get(ctx, "llm:count:"+today).Result()
	cnt, _ := strconv.Atoi(cntStr)
	if cnt >= dailyMax {
		logMsg("⚠️  LLM 今日调用已达上限")
		return ""
	}

	apiKey := getenv("ANTHROPIC_API_KEY", "")
	if apiKey == "" {
		return ""
	}

	body := map[string]interface{}{
		"model":      "claude-sonnet-4-6",
		"max_tokens": 256,
		"messages":   []map[string]string{{"role": "user", "content": prompt}},
	}
	b, _ := json.Marshal(body)
	req, _ := http.NewRequest("POST", "https://api.anthropic.com/v1/messages", strings.NewReader(string(b)))
	req.Header.Set("x-api-key", apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")
	req.Header.Set("content-type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return ""
	}
	defer resp.Body.Close()
	var cr claudeResp
	json.NewDecoder(resp.Body).Decode(&cr)

	rdb.Incr(ctx, "llm:count:"+today)
	rdb.Expire(ctx, "llm:count:"+today, 25*time.Hour)

	if len(cr.Content) > 0 {
		return strings.TrimSpace(cr.Content[0].Text)
	}
	return ""
}

// ─── Twitter OAuth 1.0a ──────────────────────────────────────────────────────

func buildTwitterOAuth(method, rawURL string, extraParams map[string]string) string {
	apiKey    := os.Getenv("TWITTER_API_KEY")
	apiSecret := os.Getenv("TWITTER_API_SECRET")
	token     := os.Getenv("TWITTER_ACCESS_TOKEN")
	tokenSecret := os.Getenv("TWITTER_ACCESS_SECRET")

	nonce := func() string {
		b := make([]byte, 16)
		rand.Read(b)
		return base64.StdEncoding.EncodeToString(b)
	}()

	ts := strconv.FormatInt(time.Now().Unix(), 10)

	params := map[string]string{
		"oauth_consumer_key":     apiKey,
		"oauth_nonce":            nonce,
		"oauth_signature_method": "HMAC-SHA1",
		"oauth_timestamp":        ts,
		"oauth_token":            token,
		"oauth_version":          "1.0",
	}
	for k, v := range extraParams {
		params[k] = v
	}

	keys := make([]string, 0, len(params))
	for k := range params {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	var parts []string
	for _, k := range keys {
		parts = append(parts, url.QueryEscape(k)+"="+url.QueryEscape(params[k]))
	}
	paramStr := strings.Join(parts, "&")
	baseStr := method + "&" + url.QueryEscape(rawURL) + "&" + url.QueryEscape(paramStr)
	sigKey := url.QueryEscape(apiSecret) + "&" + url.QueryEscape(tokenSecret)

	mac := hmac.New(sha1.New, []byte(sigKey))
	mac.Write([]byte(baseStr))
	sig := base64.StdEncoding.EncodeToString(mac.Sum(nil))
	params["oauth_signature"] = sig

	var hparts []string
	for k, v := range params {
		if strings.HasPrefix(k, "oauth_") {
			hparts = append(hparts, url.QueryEscape(k)+"=\""+url.QueryEscape(v)+"\"")
		}
	}
	return "OAuth " + strings.Join(hparts, ", ")
}

func twitterPost(text string, replyTo string) (string, error) {
	body := map[string]interface{}{"text": text}
	if replyTo != "" {
		body["reply"] = map[string]string{"in_reply_to_tweet_id": replyTo}
	}
	b, _ := json.Marshal(body)
	endpoint := "https://api.twitter.com/2/tweets"
	req, _ := http.NewRequest("POST", endpoint, strings.NewReader(string(b)))
	req.Header.Set("Authorization", buildTwitterOAuth("POST", endpoint, nil))
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 201 {
		return "", fmt.Errorf("twitter %d: %s", resp.StatusCode, string(raw))
	}
	var r struct {
		Data struct{ ID string `json:"id"` } `json:"data"`
	}
	json.Unmarshal(raw, &r)
	return r.Data.ID, nil
}

func twitterSearch(keyword string) []map[string]interface{} {
	bearer := os.Getenv("TWITTER_BEARER_TOKEN")
	if bearer == "" {
		return nil
	}
	myUID := os.Getenv("TWITTER_USER_ID")
	q := url.QueryEscape(keyword + " -is:retweet lang:en")
	u := "https://api.twitter.com/2/tweets/search/recent?query=" + q +
		"&max_results=20&tweet.fields=public_metrics,created_at,author_id"

	req, _ := http.NewRequest("GET", u, nil)
	req.Header.Set("Authorization", "Bearer "+bearer)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil
	}
	defer resp.Body.Close()
	var r struct {
		Data []struct {
			ID        string `json:"id"`
			Text      string `json:"text"`
			AuthorID  string `json:"author_id"`
			CreatedAt string `json:"created_at"`
			Metrics   struct {
				Likes int `json:"like_count"`
			} `json:"public_metrics"`
		} `json:"data"`
	}
	json.NewDecoder(resp.Body).Decode(&r)
	var out []map[string]interface{}
	for _, t := range r.Data {
		if t.AuthorID == myUID {
			continue
		}
		out = append(out, map[string]interface{}{
			"post_id":    t.ID,
			"content":    t.Text,
			"likes":      t.Metrics.Likes,
			"created_at": t.CreatedAt,
		})
	}
	return out
}

func twitterReplies(tweetID string) []map[string]string {
	bearer := os.Getenv("TWITTER_BEARER_TOKEN")
	if bearer == "" {
		return nil
	}
	u := fmt.Sprintf("https://api.twitter.com/2/tweets/search/recent?query=conversation_id:%s%%20is:reply&max_results=20&tweet.fields=author_id", tweetID)
	req, _ := http.NewRequest("GET", u, nil)
	req.Header.Set("Authorization", "Bearer "+bearer)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil
	}
	defer resp.Body.Close()
	var r struct {
		Data []struct {
			ID       string `json:"id"`
			Text     string `json:"text"`
			AuthorID string `json:"author_id"`
		} `json:"data"`
	}
	json.NewDecoder(resp.Body).Decode(&r)
	var out []map[string]string
	for _, t := range r.Data {
		out = append(out, map[string]string{"id": t.ID, "text": t.Text, "author_id": t.AuthorID})
	}
	return out
}

// ─── 币安广场 ─────────────────────────────────────────────────────────────────

func binanceSquarePost(content string) (string, error) {
	apiKey := os.Getenv("BINANCE_SQUARE_API_KEY")
	if apiKey == "" {
		return "", fmt.Errorf("BINANCE_SQUARE_API_KEY not set")
	}
	b, _ := json.Marshal(map[string]string{"bodyTextOnly": content})
	req, _ := http.NewRequest("POST",
		"https://www.binance.com/bapi/composite/v1/public/pgc/openApi/content/add",
		strings.NewReader(string(b)))
	req.Header.Set("X-Square-OpenAPI-Key", apiKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("clienttype", "binanceSkill")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)

	var r struct {
		Code    string `json:"code"`
		Message string `json:"message"`
		Data    struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	json.Unmarshal(raw, &r)
	if r.Code != "000000" {
		return "", fmt.Errorf("binance square error %s: %s", r.Code, r.Message)
	}
	return r.Data.ID, nil
}

// ─── LinkedIn ─────────────────────────────────────────────────────────────────

func linkedinPost(content string) (string, error) {
	token := os.Getenv("LINKEDIN_ACCESS_TOKEN")
	userID := os.Getenv("LINKEDIN_USER_ID")
	if token == "" || userID == "" {
		return "", fmt.Errorf("LINKEDIN_ACCESS_TOKEN or LINKEDIN_USER_ID not set")
	}
	body := map[string]interface{}{
		"author":         "urn:li:person:" + userID,
		"lifecycleState": "PUBLISHED",
		"specificContent": map[string]interface{}{
			"com.linkedin.ugc.ShareContent": map[string]interface{}{
				"shareCommentary":    map[string]string{"text": content},
				"shareMediaCategory": "NONE",
			},
		},
		"visibility": map[string]string{
			"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC",
		},
	}
	b, _ := json.Marshal(body)
	req, _ := http.NewRequest("POST", "https://api.linkedin.com/v2/ugcPosts", strings.NewReader(string(b)))
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Restli-Protocol-Version", "2.0.0")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 201 {
		return "", fmt.Errorf("linkedin %d: %s", resp.StatusCode, string(raw))
	}
	var r struct {
		ID string `json:"id"`
	}
	json.Unmarshal(raw, &r)
	return r.ID, nil
}

func linkedinComment(postURN, text string) error {
	token := os.Getenv("LINKEDIN_ACCESS_TOKEN")
	userID := os.Getenv("LINKEDIN_USER_ID")
	body := map[string]interface{}{
		"actor":   "urn:li:person:" + userID,
		"message": map[string]string{"text": text},
		"object":  postURN,
	}
	b, _ := json.Marshal(body)
	req, _ := http.NewRequest("POST", "https://api.linkedin.com/v2/socialActions/"+url.QueryEscape(postURN)+"/comments", strings.NewReader(string(b)))
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 201 {
		raw, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("linkedin comment %d: %s", resp.StatusCode, string(raw))
	}
	return nil
}

func linkedinSearch(keyword string) []map[string]interface{} {
	token := os.Getenv("LINKEDIN_ACCESS_TOKEN")
	if token == "" {
		return nil
	}
	// LinkedIn UGC search by keyword via posts search
	u := "https://api.linkedin.com/v2/ugcPosts?q=authors&authors=List()&count=20"
	req, _ := http.NewRequest("GET", u, nil)
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("X-Restli-Protocol-Version", "2.0.0")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil
	}
	defer resp.Body.Close()
	// LinkedIn doesn't have public keyword search via basic API
	// Return empty — comment engine will skip LinkedIn for now
	return nil
}

// random UA helper
func randomUA() string {
	pool := []string{
		"Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148",
		"Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 Chrome/116.0.0.0 Mobile Safari/537.36",
		"Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 Mobile/20G75",
		"Mozilla/5.0 (Linux; Android 12; SM-S906B) AppleWebKit/537.36 Chrome/112.0.0.0 Mobile Safari/537.36",
	}
	n, _ := rand.Int(rand.Reader, big.NewInt(int64(len(pool))))
	return pool[n.Int64()]
}

// ─── safety filters ──────────────────────────────────────────────────────────

var dangerWords = []string{"投诉", "骗局", "维权", "踩雷", "差评", "避坑", "黑幕", "诈骗", "虚假", "举报", "警告", "坑人", "后悔", "翻车", "赔偿", "退款", "律师", "起诉"}
var bannedWords = []string{"广告", "推广", "加微信", "私信", "关注我", "http://", "https://", "www.", "二维码"}

func isSafePost(content string) bool {
	for _, w := range dangerWords {
		if strings.Contains(content, w) {
			return false
		}
	}
	return true
}

func isSafeComment(text string) bool {
	if text == "" {
		return false
	}
	for _, w := range bannedWords {
		if strings.Contains(text, w) {
			return false
		}
	}
	return true
}

// ─── publish engine ──────────────────────────────────────────────────────────

func engineStatus() map[string]bool {
	engineMu.Lock()
	defer engineMu.Unlock()
	return map[string]bool{
		"publish": enginePublish,
		"reply":   engineReply,
		"comment": engineComment,
	}
}

func handleEngineStart(w http.ResponseWriter, r *http.Request) {
	which := r.URL.Query().Get("which") // publish | reply | comment
	engineMu.Lock()
	switch which {
	case "publish":
		enginePublish = true
		logMsg("▶️  发布引擎已启动")
	case "reply":
		engineReply = true
		logMsg("▶️  回复引擎已启动")
	case "comment":
		engineComment = true
		logMsg("▶️  评论引擎已启动")
	}
	engineMu.Unlock()
	st := engineStatus()
	broadcastEvent(Event{Ev: "engine_status", Data: st, Time: time.Now().Format("15:04:05")})
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(st)
}

func handleEngineStop(w http.ResponseWriter, r *http.Request) {
	which := r.URL.Query().Get("which")
	engineMu.Lock()
	switch which {
	case "publish":
		enginePublish = false
		logMsg("⏸️  发布引擎已暂停")
	case "reply":
		engineReply = false
		logMsg("⏸️  回复引擎已暂停")
	case "comment":
		engineComment = false
		logMsg("⏸️  评论引擎已暂停")
	}
	engineMu.Unlock()
	st := engineStatus()
	broadcastEvent(Event{Ev: "engine_status", Data: st, Time: time.Now().Format("15:04:05")})
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(st)
}

func handleEngineStatus(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(engineStatus())
}

func publishEngine() {
	time.Sleep(3 * time.Second)
	logMsg("⏰ 发布引擎就绪（等待启动）")
	for {
		engineMu.Lock()
		running := enginePublish
		engineMu.Unlock()
		if !running {
			time.Sleep(5 * time.Second)
			continue
		}
		rows, err := db.Query(`SELECT id, platform, content, retry_count FROM post_queue
			WHERE status='ready' AND publish_at <= datetime('now') LIMIT 5`)
		if err == nil {
			type qrow struct {
				ID       int64
				Platform string
				Content  string
				Retry    int
			}
			var due []qrow
			for rows.Next() {
				var r qrow
				rows.Scan(&r.ID, &r.Platform, &r.Content, &r.Retry)
				due = append(due, r)
			}
			rows.Close()

			for _, post := range due {
				logMsg(fmt.Sprintf("🚀 发布 [%s]", post.Platform))
				var postID string
				var postURL string
				var err error
				switch post.Platform {
				case "twitter":
					postID, err = twitterPost(post.Content, "")
					if err == nil {
						postURL = "https://twitter.com/i/web/status/" + postID
					}
				case "binance_square":
					postID, err = binanceSquarePost(post.Content)
					if err == nil {
						postURL = "https://www.binance.com/square/post/" + postID
					}
				case "linkedin":
					postID, err = linkedinPost(post.Content)
					if err == nil {
						postURL = "https://www.linkedin.com/feed/"
					}
				default:
					err = fmt.Errorf("unsupported platform: %s", post.Platform)
				}
				mu.Lock()
				if err == nil {
					db.Exec(`UPDATE post_queue SET status='published', post_id=?, published_at=datetime('now') WHERE id=?`, postID, post.ID)
					logMsg(fmt.Sprintf("✅ [%s] 发布成功 → %s", post.Platform, postURL))
					broadcastEvent(Event{Ev: "published", Data: map[string]string{"platform": post.Platform, "url": postURL}, Time: time.Now().Format("15:04:05")})
				} else {
					retry := post.Retry + 1
					if retry >= 3 {
						db.Exec(`UPDATE post_queue SET status='failed', error_msg=?, retry_count=? WHERE id=?`, err.Error(), retry, post.ID)
						logMsg(fmt.Sprintf("❌ [%s] 彻底失败（3次）: %v", post.Platform, err))
					} else {
						delays := []string{"1 minute", "5 minutes", "15 minutes"}
						db.Exec(fmt.Sprintf(`UPDATE post_queue SET publish_at=datetime('now','+%s'), retry_count=?, error_msg=? WHERE id=?`, delays[retry-1]), retry, err.Error(), post.ID)
						logMsg(fmt.Sprintf("⚠️  [%s] 失败，稍后重试（第%d次）", post.Platform, retry))
					}
				}
				mu.Unlock()
			}
		}
		time.Sleep(30 * time.Second)
	}
}

// ─── reply engine ────────────────────────────────────────────────────────────

func replyEngine() {
	time.Sleep(5 * time.Second)
	logMsg("💬 回复引擎就绪（等待启动）")
	myUID := os.Getenv("TWITTER_USER_ID")
	brandTone := getenv("BRAND_TONE", "专业、友好、有价值感")

	for {
		engineMu.Lock()
		running := engineReply
		engineMu.Unlock()
		if !running {
			time.Sleep(5 * time.Second)
			continue
		}
		rows, _ := db.Query(`SELECT post_id, platform FROM post_queue
			WHERE status='published' AND published_at >= datetime('now','-7 days') AND post_id IS NOT NULL`)
		var posts []struct{ PostID, Platform string }
		for rows.Next() {
			var p struct{ PostID, Platform string }
			rows.Scan(&p.PostID, &p.Platform)
			posts = append(posts, p)
		}
		rows.Close()

		for _, post := range posts {
			var comments []map[string]string
			switch post.Platform {
			case "twitter":
				comments = twitterReplies(post.PostID)
			case "binance_square", "linkedin":
				// 暂不支持评论读取 API
				comments = nil
			}
			for _, c := range comments {
				authorID := c["author_id"]
				if post.Platform == "twitter" && myUID != "" && authorID == myUID {
					continue
				}
				var exists int
				db.QueryRow(`SELECT COUNT(*) FROM reply_log WHERE platform=? AND comment_id=?`, post.Platform, c["id"]).Scan(&exists)
				if exists > 0 {
					continue
				}

				reply := llmAsk(fmt.Sprintf(`Brand tone: %s
User comment: %s

Reply naturally (like a real person), max 50 words, no links or product mentions. Return only the reply.`, brandTone, c["text"]))

				status := "failed"
				if reply != "" && isSafeComment(reply) {
					var replyErr error
					switch post.Platform {
					case "twitter":
						_, replyErr = twitterPost(reply, c["id"])
					default:
						replyErr = fmt.Errorf("reply not supported for platform: %s", post.Platform)
					}
					if replyErr == nil {
						status = "success"
						logMsg(fmt.Sprintf("↩️  [%s] 回复完成", post.Platform))
						broadcastEvent(Event{Ev: "replied", Data: map[string]string{"platform": post.Platform, "reply": reply}, Time: time.Now().Format("15:04:05")})
					}
				}
				mu.Lock()
				db.Exec(`INSERT INTO reply_log (platform,post_id,comment_id,comment_text,reply_text,status,replied_at,created_at)
					VALUES (?,?,?,?,?,?,datetime('now'),datetime('now'))`,
					post.Platform, post.PostID, c["id"], c["text"], reply, status)
				mu.Unlock()
				randomSleep(8, 20)
			}
		}

		wait := randomSleep(120, 300)
		logMsg(fmt.Sprintf("⏳ 回复引擎下次轮询等待 %d 秒", wait))
	}
}

// ─── comment engine ──────────────────────────────────────────────────────────

func commentEngine() {
	time.Sleep(8 * time.Second)
	logMsg("🔍 评论引擎就绪（等待启动）")

	for {
		engineMu.Lock()
		running := engineComment
		engineMu.Unlock()
		if !running {
			time.Sleep(5 * time.Second)
			continue
		}
		dailyMax, _ := strconv.Atoi(getenv("COMMENT_DAILY_LIMIT", "30"))
		today := todayPrefix()
		var todayCount int
		db.QueryRow(`SELECT COUNT(*) FROM comment_log WHERE status='success' AND posted_at LIKE ?`, today+"%").Scan(&todayCount)

		if todayCount >= dailyMax {
			logMsg(fmt.Sprintf("⏸️  今日评论已达上限（%d条）", dailyMax))
			time.Sleep(time.Hour)
			continue
		}

		platforms := strings.Split(getenv("TARGET_PLATFORMS", "twitter"), ",")
		platform := strings.TrimSpace(platforms[int(time.Now().Unix()%int64(len(platforms)))])
		keywords := strings.Split(getenv("COMMENT_KEYWORDS", "AI tools"), ",")
		keyword := strings.TrimSpace(keywords[int(time.Now().Unix()/13%int64(len(keywords)))])
		logMsg(fmt.Sprintf("🔍 [%s] 搜索: %s", platform, keyword))

		var posts []map[string]interface{}
		switch platform {
		case "twitter":
			posts = twitterSearch(keyword)
		case "linkedin":
			posts = linkedinSearch(keyword)
		default:
			// binance_square 暂无公开搜索 API
			posts = nil
		}
		if len(posts) == 0 {
			time.Sleep(60 * time.Second)
			continue
		}

		// filter already commented
		rows, _ := db.Query(`SELECT target_post_id FROM comment_log WHERE platform=? AND status='success'`, platform)
		done := map[string]bool{}
		for rows.Next() {
			var id string
			rows.Scan(&id)
			done[id] = true
		}
		rows.Close()

		var target map[string]interface{}
		for _, p := range posts {
			pid := fmt.Sprint(p["post_id"])
			likes, _ := p["likes"].(int)
			if !done[pid] && likes > 10 && isSafePost(fmt.Sprint(p["content"])) {
				target = p
				break
			}
		}
		if target == nil {
			logMsg("⚠️  本批帖子均已评论，等待30分钟")
			time.Sleep(30 * time.Minute)
			continue
		}

		// 优先用模板库，没有模板才调 LLM
		var comment string
		templates := os.Getenv("COMMENT_TEMPLATES")
		if templates != "" {
			tpls := strings.Split(templates, "\n")
			var valid []string
			for _, t := range tpls {
				t = strings.TrimSpace(t)
				if t != "" {
					valid = append(valid, t)
				}
			}
			if len(valid) > 0 {
				comment = valid[int(time.Now().UnixNano()%int64(len(valid)))]
			}
		}
		if comment == "" {
			// 没有模板，用 LLM 生成
			styleMap := map[string]string{
				"有价值的补充": "Add a useful piece of information not mentioned in the post, max 50 words.",
				"提问互动":    "Ask a specific genuine question, max 30 words.",
				"经验分享":    "Share a brief relevant personal experience, max 60 words.",
			}
			style := styleMap[getenv("COMMENT_STYLE", "有价值的补充")]
			comment = llmAsk(fmt.Sprintf(`Post: %s

Task: %s

Rules: sound like a real person, no links, no promotions, no @mentions. Return only the comment.`,
				fmt.Sprint(target["content"]), style))
		}

		if comment == "" || !isSafeComment(comment) {
			logMsg("⚠️  评论生成失败或被过滤，跳过")
			time.Sleep(30 * time.Second)
			continue
		}
		logMsg(fmt.Sprintf("✍️  评论: %s", comment))

		postID := fmt.Sprint(target["post_id"])
		mu.Lock()
		var recID int64
		db.QueryRow(`INSERT INTO comment_log (platform,target_post_id,keyword,comment_text,status,created_at)
			VALUES ('twitter',?,?,?,'pending',datetime('now')) RETURNING id`, postID, keyword, comment).Scan(&recID)
		mu.Unlock()

		var commentErr error
		switch platform {
		case "twitter":
			_, commentErr = twitterPost(comment, postID)
		case "linkedin":
			commentErr = linkedinComment(postID, comment)
		default:
			commentErr = fmt.Errorf("comment not supported for platform: %s", platform)
		}
		status := "success"
		errMsg := ""
		if commentErr != nil {
			status = "failed"
			errMsg = commentErr.Error()
			logMsg(fmt.Sprintf("❌ [%s] 评论失败: %v", platform, commentErr))
		} else {
			logMsg(fmt.Sprintf("✅ [%s] 评论成功（今日第%d条）", platform, todayCount+1))
			broadcastEvent(Event{Ev: "commented", Data: map[string]string{"platform": platform, "keyword": keyword, "comment": comment}, Time: time.Now().Format("15:04:05")})
		}

		mu.Lock()
		postedAt := ""
		if status == "success" {
			postedAt = time.Now().UTC().Format("2006-01-02 15:04:05")
		}
		db.Exec(`UPDATE comment_log SET status=?, posted_at=?, error_msg=? WHERE id=?`, status, postedAt, errMsg, recID)
		mu.Unlock()

		minSec, _ := strconv.Atoi(getenv("COMMENT_INTERVAL_MIN", "180"))
		maxSec, _ := strconv.Atoi(getenv("COMMENT_INTERVAL_MAX", "600"))
		wait := randomSleep(minSec, maxSec)
		logMsg(fmt.Sprintf("⏳ 等待 %d 秒...", wait))
	}
}

// ─── log worker ──────────────────────────────────────────────────────────────

var recentLogs []string
var logMu sync.Mutex

func logWorker() {
	f, _ := os.OpenFile("./logs/autopilot.log", os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
	for line := range logCh {
		if f != nil {
			f.WriteString(line + "\n")
		}
		logMu.Lock()
		recentLogs = append([]string{line}, recentLogs...)
		if len(recentLogs) > 200 {
			recentLogs = recentLogs[:200]
		}
		logMu.Unlock()
	}
}

// ─── snapshot ────────────────────────────────────────────────────────────────

func getSnapshot() map[string]interface{} {
	today := todayPrefix()

	var qReady, qPublished, qFailed int
	db.QueryRow(`SELECT COUNT(*) FROM post_queue WHERE status='ready'`).Scan(&qReady)
	db.QueryRow(`SELECT COUNT(*) FROM post_queue WHERE status='published'`).Scan(&qPublished)
	db.QueryRow(`SELECT COUNT(*) FROM post_queue WHERE status='failed'`).Scan(&qFailed)

	var cToday, cTotal, rToday, rTotal int
	db.QueryRow(`SELECT COUNT(*) FROM comment_log WHERE status='success' AND posted_at LIKE ?`, today+"%").Scan(&cToday)
	db.QueryRow(`SELECT COUNT(*) FROM comment_log WHERE status='success'`).Scan(&cTotal)
	db.QueryRow(`SELECT COUNT(*) FROM reply_log WHERE status='success' AND replied_at LIKE ?`, today+"%").Scan(&rToday)
	db.QueryRow(`SELECT COUNT(*) FROM reply_log WHERE status='success'`).Scan(&rTotal)

	llmCnt, _ := rdb.Get(ctx, "llm:count:"+today).Result()
	llmMax, _ := strconv.Atoi(getenv("LLM_DAILY_LIMIT", "200"))

	llmN, _ := strconv.Atoi(llmCnt)

	return map[string]interface{}{
		"queue":    map[string]int{"ready": qReady, "published": qPublished, "failed": qFailed},
		"comments": map[string]int{"today": cToday, "total": cTotal},
		"replies":  map[string]int{"today": rToday, "total": rTotal},
		"llm":      map[string]interface{}{"today": llmN, "daily_max": llmMax},
	}
}

// ─── HTTP handlers ───────────────────────────────────────────────────────────

func serveHTML(w http.ResponseWriter, r *http.Request) {
	http.ServeFile(w, r, "./dashboard.html")
}

func serveWS(w http.ResponseWriter, r *http.Request) {
	// Simple SSE instead of WebSocket for easier deployment
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	ch := make(chan string, 64)
	wsMu.Lock()
	wsConns = append(wsConns, ch)
	wsMu.Unlock()

	defer func() {
		wsMu.Lock()
		for i, c := range wsConns {
			if c == ch {
				wsConns = append(wsConns[:i], wsConns[i+1:]...)
				break
			}
		}
		wsMu.Unlock()
	}()

	snap, _ := json.Marshal(Event{Ev: "snapshot", Data: getSnapshot(), Time: time.Now().Format("15:04:05")})
	fmt.Fprintf(w, "data: %s\n\n", snap)
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}

	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case msg := <-ch:
			fmt.Fprintf(w, "data: %s\n\n", msg)
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
		case <-ticker.C:
			snap, _ := json.Marshal(Event{Ev: "snapshot", Data: getSnapshot(), Time: time.Now().Format("15:04:05")})
			fmt.Fprintf(w, "data: %s\n\n", snap)
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
		case <-r.Context().Done():
			return
		}
	}
}

func handleQueueAdd(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "method not allowed", 405)
		return
	}
	var body struct {
		Platform  string `json:"platform"`
		Content   string `json:"content"`
		PublishAt string `json:"publish_at"`
	}
	json.NewDecoder(r.Body).Decode(&body)
	if body.Platform == "" || body.Content == "" || body.PublishAt == "" {
		http.Error(w, `{"error":"missing fields"}`, 400)
		return
	}
	mu.Lock()
	db.Exec(`INSERT INTO post_queue (platform,content,publish_at,created_at) VALUES (?,?,?,datetime('now'))`,
		body.Platform, body.Content, body.PublishAt)
	mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]bool{"ok": true})
}

func handleQueueList(w http.ResponseWriter, r *http.Request) {
	rows, _ := db.Query(`SELECT id,platform,content,publish_at,status,post_id,error_msg,retry_count,created_at,published_at
		FROM post_queue ORDER BY publish_at DESC LIMIT 50`)
	defer rows.Close()
	var list []map[string]interface{}
	for rows.Next() {
		var id, retry int64
		var platform, content, publishAt, status, createdAt string
		var postID, errMsg, publishedAt sql.NullString
		rows.Scan(&id, &platform, &content, &publishAt, &status, &postID, &errMsg, &retry, &createdAt, &publishedAt)
		list = append(list, map[string]interface{}{
			"id": id, "platform": platform, "content": content,
			"publish_at": publishAt, "status": status,
			"post_id": postID.String, "error_msg": errMsg.String,
			"retry_count": retry, "created_at": createdAt,
			"published_at": publishedAt.String,
		})
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(list)
}

func handleSnapshot(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(getSnapshot())
}

func handleLogs(w http.ResponseWriter, r *http.Request) {
	logMu.Lock()
	logs := make([]string, len(recentLogs))
	copy(logs, recentLogs)
	logMu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(logs)
}

// handleConfig — GET 返回当前配置（脱敏），POST 保存
func handleConfig(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	mask := func(v string) string {
		if len(v) > 8 {
			return v[:4] + strings.Repeat("*", len(v)-8) + v[len(v)-4:]
		}
		if v != "" {
			return "****"
		}
		return ""
	}
	json.NewEncoder(w).Encode(map[string]string{
		"TWITTER_API_KEY":         mask(os.Getenv("TWITTER_API_KEY")),
		"TWITTER_API_SECRET":      mask(os.Getenv("TWITTER_API_SECRET")),
		"TWITTER_ACCESS_TOKEN":    mask(os.Getenv("TWITTER_ACCESS_TOKEN")),
		"TWITTER_ACCESS_SECRET":   mask(os.Getenv("TWITTER_ACCESS_SECRET")),
		"TWITTER_BEARER_TOKEN":    mask(os.Getenv("TWITTER_BEARER_TOKEN")),
		"TWITTER_USER_ID":         os.Getenv("TWITTER_USER_ID"),
		"BINANCE_SQUARE_API_KEY":  mask(os.Getenv("BINANCE_SQUARE_API_KEY")),
		"LINKEDIN_ACCESS_TOKEN":   mask(os.Getenv("LINKEDIN_ACCESS_TOKEN")),
		"LINKEDIN_USER_ID":        os.Getenv("LINKEDIN_USER_ID"),
		"TARGET_PLATFORMS":        getenv("TARGET_PLATFORMS", "twitter"),
		"BRAND_NAME":              os.Getenv("BRAND_NAME"),
		"BRAND_TONE":              os.Getenv("BRAND_TONE"),
		"COMMENT_KEYWORDS":        os.Getenv("COMMENT_KEYWORDS"),
		"COMMENT_TEMPLATES":       os.Getenv("COMMENT_TEMPLATES"),
		"COMMENT_DAILY_LIMIT":     getenv("COMMENT_DAILY_LIMIT", "30"),
		"LLM_DAILY_LIMIT":         getenv("LLM_DAILY_LIMIT", "200"),
		"NOTIFY_WEBHOOK":          os.Getenv("NOTIFY_WEBHOOK"),
	})
}

func handleConfigSave(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "method not allowed", 405)
		return
	}
	var body map[string]string
	json.NewDecoder(r.Body).Decode(&body)
	// always allow TARGET_PLATFORMS even if it doesn't contain ****


	// read current env file
	lines := []string{}
	if raw, err := os.ReadFile("./autopilot.env"); err == nil {
		lines = strings.Split(string(raw), "\n")
	}

	// update or append each key
	updated := map[string]bool{}
	for i, line := range lines {
		for k, v := range body {
			if v == "" || strings.Contains(v, "****") {
				continue // don't overwrite with masked values
			}
			if strings.HasPrefix(line, k+"=") {
				lines[i] = k + "=" + v
				updated[k] = true
				os.Setenv(k, v)
			}
		}
	}
	for k, v := range body {
		if v == "" || strings.Contains(v, "****") {
			continue
		}
		if !updated[k] {
			lines = append(lines, k+"="+v)
			os.Setenv(k, v)
		}
	}

	os.WriteFile("./autopilot.env", []byte(strings.Join(lines, "\n")), 0600)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]bool{"ok": true})
}
