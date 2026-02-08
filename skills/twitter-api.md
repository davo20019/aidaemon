---
name: twitter-api
description: X/Twitter API knowledge, error diagnosis, and tier limitations
triggers: tweet, twitter, x.com, reply, comment, post, retweet, like, follow, mention, hashtag, timeline
---
When working with the X (Twitter) API, use this knowledge to diagnose errors and avoid loops.

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
| "Forbidden" (generic) | App permissions set to "Read" only | Change to "Read and Write" in X Developer Portal, then re-do OAuth flow |
| 403 after changing permissions | Old OAuth token has stale scopes | Remove and reconnect OAuth: `manage_oauth` remove twitter, then connect |

### Replying to a Tweet

To reply, include the `reply` field:
```json
POST https://api.twitter.com/2/tweets
{
  "text": "your reply text",
  "reply": {
    "in_reply_to_tweet_id": "TWEET_ID_HERE"
  }
}
```
**IMPORTANT**: This only works if the API can "see" the target tweet. On Free tier, you can only reply to your own tweets.

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
- After changing app permissions in X Developer Portal, you MUST remove and reconnect the OAuth connection to get a new token with updated scopes
