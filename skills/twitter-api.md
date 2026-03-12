---
name: twitter-api
description: X/Twitter API knowledge, error diagnosis, and tier limitations
triggers: twitter api, x api, twitter oauth, twitter auth, twitter developer, twitter error, x error, tweet api, tweet visibility, twitter timeline, post to twitter, post to x, twitter post, tweet this, reply to tweet, quote tweet
---
When working with the X (Twitter) API, use this knowledge to diagnose errors and avoid loops.

### Common Actions

Use the `http_request` tool with `auth_profile="twitter"`.

Create a post:
```json
POST https://api.x.com/2/tweets
{
  "text": "your message"
}
```

Reply to a post:
```json
POST https://api.x.com/2/tweets
{
  "text": "your reply text",
  "reply": {
    "in_reply_to_tweet_id": "TWEET_ID_HERE"
  }
}
```

Quote a post:
```json
POST https://api.x.com/2/tweets
{
  "text": "your comment",
  "quote_tweet_id": "TWEET_ID_HERE"
}
```

### API Tier Limitations (IMPORTANT)

**Free Tier** ($0/month):
- POST /2/tweets — post tweets (1,500/month) ✓
- DELETE /2/tweets/:id — delete own tweets ✓
- Reply to your OWN tweets ✓
- **CANNOT** read/lookup other users' tweets
- **CANNOT** reply to other users' tweets (403: "not visible to you")
- **CANNOT** search tweets, get timelines, or lookup users
- **CANNOT** use streaming endpoints

**Basic Tier** ($200/month):
- Everything in Free, plus:
- GET /2/tweets/:id — read any public tweet
- Reply to any public tweet
- GET /2/users — user lookup
- 10,000 tweet reads/month, 3,000 posts/month

**Pro Tier** ($5,000/month):
- Full API access, search, streaming, analytics

### Common 403 Errors and What They Mean

| Error Message | Cause | Fix |
|---|---|---|
| "You attempted to reply to a Tweet that is deleted or not visible to you" | Free tier can't see other users' tweets | Need Basic tier ($200/mo) to reply to others |
| "You are not allowed to create a Tweet with duplicate content" | Exact same tweet text posted recently | Change the text slightly |
| "Forbidden" (generic) | App permissions set to "Read" only | Change to "Read and Write" in X Developer Portal, then start a fresh OAuth connect flow |
| 403 after changing permissions | Old OAuth token has stale scopes | Run `manage_oauth` connect for twitter again. Do not remove the existing connection first |

**IMPORTANT**: Replying only works if the API can "see" the target tweet. On Free tier, you can only reply to your own tweets.

### Error Handling Rules

1. **NEVER retry the same failing request more than once.** If you get a 403, diagnose it — don't loop.
2. If the error says "not visible to you", explain the tier limitation to the user. Don't try to reconnect OAuth — it won't help.
3. If the error says "duplicate content", modify the tweet text and try once more.
4. If generic 403 after OAuth reconnect, check the app permissions at developer.x.com.
5. **Always show the user the actual error message from X** so they can take action.

### OAuth Notes

- aidaemon uses OAuth 2.0 PKCE for X authentication
- Scopes requested: tweet.read, tweet.write, users.read, offline.access
- Tokens stored in OS keychain as `oauth_twitter_access_token`
- After changing app permissions in X Developer Portal, start a new OAuth connect flow to get a token with updated scopes
- Do not remove the old Twitter connection first; if reconnect fails or times out, you want the current connection to remain available
