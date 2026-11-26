# 美股推播：外部準時觸發設定（cron-job.org）

GitHub 內建排程常延遲數小時，改由外部服務準時呼叫 GitHub；內建 cron 留作備援。
同日同班（開盤半小時／收盤）只推一次：先到的推，後到的看到 `tracking/us_push_sent.txt` 已記錄就跳過。
保留既有保護：休市日不推、開盤半小時快報超過美東 11:00 不推。

## 建立 cron-job.org 排程（共用 token，台股設定見 tw-breakout-radar/tw-push/external-trigger-setup.md）
- URL：`https://api.github.com/repos/Joesung0719/pro-terminal/actions/workflows/us_push.yml/dispatches`
- Request method：POST
- Headers：
  - `Authorization: Bearer <你的 token>`
  - `Accept: application/vnd.github+json`
  - `X-GitHub-Api-Version: 2022-11-28`
  - `Content-Type: application/json`
  - `User-Agent: cron-job-org`
- **Time zone：America/New_York**（夏令／冬令自動切換，不用每年改）；週一至週五
- 預期回應 HTTP 204 為成功

| 排程 | 美東時間 | 台北（冬令） | Body |
|---|---|---|---|
| 開盤半小時 | 10:05 | 23:05 | `{"ref":"main","inputs":{"slot":"open"}}` |
| 收盤 | 16:20 | 隔日 05:20 | `{"ref":"main","inputs":{"slot":"close"}}` |

## 驗證
1. Test run 回 204
2. GitHub → Actions，名稱為 `us-line-push open`／`us-line-push close`
3. 備援 cron 延遲跑到時，日誌出現「已推過，本次不重複推播」
