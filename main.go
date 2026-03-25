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
	logMsg("   引流关键词: " + getenv("COMMENT_KEYWORDS", ""))

	http.HandleFunc("/", serveHTML)
	http.HandleFunc("/ws", serveWS)
	http.HandleFunc("/api/queue/add", handleQueueAdd)
	http.HandleFunc("/api/queue", handleQueueList)
	http.HandleFunc("/api/snapshot", handleSnapshot)
	http.HandleFunc("/api/config", handleConfig)
	http.HandleFunc("/api/config/save", handleConfigSave)
	http.HandleFunc("/api/logs", handleLogs)

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

func publishEngine() {
	time.Sleep(3 * time.Second)
	logMsg("⏰ 发布引擎启动")
	for {
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
				tweetID, err := twitterPost(post.Content, "")
				mu.Lock()
				if err == nil {
					db.Exec(`UPDATE post_queue SET status='published', post_id=?, published_at=datetime('now') WHERE id=?`, tweetID, post.ID)
					logMsg(fmt.Sprintf("✅ [%s] 发布成功 → https://twitter.com/i/web/status/%s", post.Platform, tweetID))
					broadcastEvent(Event{Ev: "published", Data: map[string]string{"platform": post.Platform, "url": "https://twitter.com/i/web/status/" + tweetID}, Time: time.Now().Format("15:04:05")})
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
	logMsg("💬 自动回复引擎启动")
	myUID := os.Getenv("TWITTER_USER_ID")
	brandTone := getenv("BRAND_TONE", "专业、友好、有价值感")

	for {
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
			comments := twitterReplies(post.PostID)
			for _, c := range comments {
				if myUID != "" && c["author_id"] == myUID {
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
					_, err := twitterPost(reply, c["id"])
					if err == nil {
						status = "success"
						logMsg(fmt.Sprintf("↩️  [twitter] 回复完成"))
						broadcastEvent(Event{Ev: "replied", Data: map[string]string{"reply": reply}, Time: time.Now().Format("15:04:05")})
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
	logMsg("🔍 评论引流引擎启动")

	for {
		dailyMax, _ := strconv.Atoi(getenv("COMMENT_DAILY_LIMIT", "30"))
		today := todayPrefix()
		var todayCount int
		db.QueryRow(`SELECT COUNT(*) FROM comment_log WHERE status='success' AND posted_at LIKE ?`, today+"%").Scan(&todayCount)

		if todayCount >= dailyMax {
			logMsg(fmt.Sprintf("⏸️  今日评论已达上限（%d条）", dailyMax))
			time.Sleep(time.Hour)
			continue
		}

		keywords := strings.Split(getenv("COMMENT_KEYWORDS", "AI tools"), ",")
		keyword := strings.TrimSpace(keywords[int(time.Now().Unix()/13)%len(keywords)])
		logMsg(fmt.Sprintf("🔍 [twitter] 搜索: %s", keyword))

		posts := twitterSearch(keyword)
		if len(posts) == 0 {
			time.Sleep(60 * time.Second)
			continue
		}

		// filter already commented
		rows, _ := db.Query(`SELECT target_post_id FROM comment_log WHERE platform='twitter' AND status='success'`)
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

		styleMap := map[string]string{
			"有价值的补充": "Add a useful piece of information not mentioned in the post, max 50 words.",
			"提问互动":    "Ask a specific genuine question, max 30 words.",
			"经验分享":    "Share a brief relevant personal experience, max 60 words.",
		}
		style := styleMap[getenv("COMMENT_STYLE", "有价值的补充")]
		comment := llmAsk(fmt.Sprintf(`Post: %s

Task: %s

Rules: sound like a real person, no links, no promotions, no @mentions. Return only the comment.`,
			fmt.Sprint(target["content"]), style))

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

		_, err := twitterPost(comment, postID)
		status := "success"
		errMsg := ""
		if err != nil {
			status = "failed"
			errMsg = err.Error()
			logMsg(fmt.Sprintf("❌ 评论失败: %v", err))
		} else {
			logMsg(fmt.Sprintf("✅ 评论成功（今日第%d条）", todayCount+1))
			broadcastEvent(Event{Ev: "commented", Data: map[string]string{"keyword": keyword, "comment": comment}, Time: time.Now().Format("15:04:05")})
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
		"TWITTER_API_KEY":       mask(os.Getenv("TWITTER_API_KEY")),
		"TWITTER_API_SECRET":    mask(os.Getenv("TWITTER_API_SECRET")),
		"TWITTER_ACCESS_TOKEN":  mask(os.Getenv("TWITTER_ACCESS_TOKEN")),
		"TWITTER_ACCESS_SECRET": mask(os.Getenv("TWITTER_ACCESS_SECRET")),
		"TWITTER_BEARER_TOKEN":  mask(os.Getenv("TWITTER_BEARER_TOKEN")),
		"TWITTER_USER_ID":       os.Getenv("TWITTER_USER_ID"),
		"BRAND_NAME":            os.Getenv("BRAND_NAME"),
		"BRAND_TONE":            os.Getenv("BRAND_TONE"),
		"COMMENT_KEYWORDS":      os.Getenv("COMMENT_KEYWORDS"),
		"COMMENT_DAILY_LIMIT":   getenv("COMMENT_DAILY_LIMIT", "30"),
		"LLM_DAILY_LIMIT":       getenv("LLM_DAILY_LIMIT", "200"),
		"NOTIFY_WEBHOOK":        os.Getenv("NOTIFY_WEBHOOK"),
	})
}

func handleConfigSave(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "method not allowed", 405)
		return
	}
	var body map[string]string
	json.NewDecoder(r.Body).Decode(&body)

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
