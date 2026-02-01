---
name: job-search
description: Help the user search for jobs on LinkedIn, Indeed, and other job boards
triggers: job, jobs, career, hiring, linkedin, indeed, glassdoor, resume, interview
---
When the user asks about job searching, follow this approach:

### Before Searching — Ask the User
1. What role/title are they looking for?
2. What location (or remote)?
3. Any salary range or company preferences?
4. Experience level (entry, mid, senior)?

### URL Patterns

**LinkedIn Jobs:**
`https://www.linkedin.com/jobs/search/?keywords=<role>&location=<location>`

**Indeed:**
`https://www.indeed.com/jobs?q=<role>&l=<location>`

**Glassdoor:**
`https://www.glassdoor.com/Job/jobs.htm?sc.keyword=<role>&locT=C&locKeyword=<location>`

### Workflow
1. Use the `browser` tool to navigate to the job board URL.
2. Screenshot the results page.
3. Extract job listings using get_text.
4. Present results in a structured format:
   - **Title** — Company — Location
   - Salary (if shown)
   - Link to the posting

### Rules
- Always use the `browser` tool for job sites, not curl/wget.
- If the user is logged into LinkedIn via their Chrome profile, prefer LinkedIn for better results.
- Summarize the top 5-10 results rather than dumping raw text.
- Offer to dig deeper into specific postings if the user is interested.
