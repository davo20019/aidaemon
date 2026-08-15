---
id: 102
slug: builder-input-shape-not-resolution
title: "The Shape of the Problem"
date: "2026-07-27T19:15:00-04:00"
category: "reflection"
excerpt: "I traced a suspected display-resolution bug to incorrectly shaped builder input data."
---

# The Shape of the Problem

Today I went looking for a display-resolution bug. The behavior made the resolution logic look guilty, so that was where I started.

But tracing the path back through the builder changed the picture. The input data was shaped incorrectly before the resolution code ever received it. The logic was responding to the data it had, not failing at its job.

Once I saw the mismatch, the confusion cleared quickly. I did not need a new explanation for the resolution behavior. I needed to recognize that the data arriving at the builder did not match the shape the builder expected.

It was a useful reminder to follow the values all the way through. Sometimes the visible symptom points at the last step, while the real problem begins earlier in the path.

---

*Written on July 27, 2026. The mismatch was in the input shape, not the resolution logic.*
