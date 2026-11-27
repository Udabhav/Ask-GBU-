# Demo script (about 2 minutes)

A script for the Phase 6 demo video. Every query below was checked against the app (knowledge base v0.2, threshold 0.69) and gives the result shown. If you change the knowledge base, check the queries again before recording.

## Before recording

1. Run `python -m streamlit run app.py` and wait for the page to load. The first question after a start can take a few seconds.
2. Click **Clear chat** in the sidebar.
3. Zoom the browser to about 125% so the text is readable in the video.
4. Open `eval/results/threshold_sweep.png` in another tab for the last part.

## Script

| Time | Do | Say |
|---|---|---|
| 0:00 to 0:15 | Show the page | "Ask GBU is an FAQ chatbot for Gautam Buddha University. It answers only from facts checked against official GBU sources, in English, Hinglish, and Hindi." |
| 0:15 to 0:35 | Type: **How much does B.Tech CSE cost per semester?** | "A normal question. It answers Rs. 1,10,000 per semester, with the academic year and a link to the official brochure." |
| 0:35 to 0:55 | Type: **admission abhi chal rahe hai kya** | "The same in Hinglish. It says admissions for 2026-27 are closed." Open **How was this matched?** "The intent classifier detected the admissions topic, so the bot searched only admission records." |
| 0:55 to 1:10 | Type: **लड़कियों को पहले साल में कौन सा कमरा मिलता है** | "And in Hindi: girls get a double room in the first year, and a single room from the second year." |
| 1:10 to 1:25 | Type: **who won the cricket match yesterday** | "An off-topic question. The classifier knows it is not about GBU, so the bot refuses instead of guessing." |
| 1:25 to 1:40 | Type: **MBA ki fees kitni hai** | "A GBU question we have no verified data for. The closest match scores below the threshold, so it refuses instead of showing the B.Tech fee." |
| 1:40 to 2:00 | Show `threshold_sweep.png` | "We tested it on 69 unseen questions. It gives the right answer or rightly refuses for 74% of them, and gives a wrong answer for about 10%. We tuned this threshold to keep wrong answers low." |

## Expected results

| Query | Expected result |
|---|---|
| How much does B.Tech CSE cost per semester? | B.Tech CSE fee (score about 0.88) |
| admission abhi chal rahe hai kya | Admissions closed (topic: admissions) |
| लड़कियों को पहले साल में कौन सा कमरा मिलता है | Girls' room type (score about 0.77) |
| who won the cricket match yesterday | Fallback: not about GBU |
| MBA ki fees kitni hai | Fallback: low score (about 0.58) |

Avoid **hostel mein mess ka kitna lagta hai** in the demo. It is a known failure and returns the hostel allotment answer.
