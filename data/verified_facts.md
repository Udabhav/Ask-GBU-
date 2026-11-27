# Verified facts (Ask GBU)

Only facts listed here are verified from official Gautam Buddha University (gbu.ac.in) sources. Anything not listed is unverified and must not go into the knowledge base yet.

**Primary source:** Admission Brochure 2026-27 (v1), `https://www.gbu.ac.in/content/gbudata/Admission/Admission-Brochure_2026-27-v1.pdf`
Text extracted to `data/raw/brochure_2026-27-v1_extracted.txt`. Checked on 2026-11-27.

**Hostel source:** `https://hostels.gbu.ac.in/abouthostels.php`, checked on 2026-11-27.

---

## Verified

### Fees (2026-27, Indian nationals, Brochure section 7.1)
| Item | Amount | Notes |
|---|---|---|
| Counseling cum Registration Fee | Rs. 10,000 | One time |
| Security Fee | Rs. 10,000 | One time, refundable |
| Library Fee | Rs. 750 | Per semester |
| Sports Fee | Rs. 2,500 | Per semester |
| B.Tech. Computer Science and Engineering | Rs. 1,10,000 | Academic fee, per semester |

**Caution:** The fee table text was extracted from a PDF, and some rows are misaligned. Only the B.Tech CSE row above is confirmed. Check every other program's fee against the PDF before using it.

### Hostel and mess (Brochure section D, per academic year)
| Item | Amount |
|---|---|
| Single seated room | Rs. 50,000 |
| Double seated room | Rs. 30,000 |
| Married Research Scholars' Hostel | Rs. 60,000 |
| Mess charges | Rs. 40,000 (tentative, can change with the tender) |

### Hostels (Brochure section 2.1 and hostel website)
- The brochure says: 11 boys' hostels and 7 girls' hostels.
- **Confirmed by user (2026-11-27):** total girls' hostels = 7. This matches the brochure.
- The hostel website lists 11 boys' hostels, 6 girls' hostels, and the Married Research Scholars Hostel. The website count is older or incomplete, so use 7 for girls' hostels.
- Girls' hostels: double occupancy in the first year, single occupancy from the second year (brochure).
- Hostel allotment is "first come, first served" (brochure). Allotment and exemptions are handled through the Online Hostel Management System (OHMS).
- Facilities on the website: TT room, TV room, reading room, first aid room, indoor sports, 24x7 ambulance, dispensary, four meals a day, and round-the-clock security.

### Admissions (provided by user, 2026-11-27; check on gbu.ac.in before publishing)
- **Status:** Admissions for the 2026-27 session are **closed**. The bot must say so, and must not tell students to apply.
- **Admission site:** `https://gbuadm.samarth.edu.in/`
- **Admission phone:** 0120-2344234 or 0120-2344247
- **Admission email:** `admissions@gbu.ac.in`

### Contacts
- **Hostels:** Phone 0120-234 6175, email `hostels@gbu.ac.in` (hostels website).
- **International Affairs:** `director_ia@gbu.ac.in`, phone 120-2346179 and 120-2344250, mobile +91-8130084839 (brochure).
- **Address:** Gautam Buddha University, Greater Noida, Gautam Buddha Nagar, Uttar Pradesh 201312, India (brochure).

### Scholarships (Brochure section 8.0)
- SC/ST students: 50% scholarship on tuition fees (rest may be covered under the UP scheme).
- Annual merit scholarship for top-ranked students.
- New Entrant Scholarship.
- Full details, eligibility, and deadlines are in section 8.0. Read it before writing any scholarship answer.

---

## Not verified yet (do not use)

- **Exact admission dates.** Admissions are closed, so the exact last date is no longer needed for students. Only state "closed" unless a new session's notice is published.
- **Admission contacts and portal.** Provided by the user (see above), but not yet checked on the official site. Add the source link when verified.
- **Fees for all programs except B.Tech CSE.** Re-check the table.
- **B.Tech eligibility criteria.** Available in the brochure but not yet written into records.
- **Revised brochure (1 July 2026).** `Admission Brochure-revised-1July2026.pdf` was too large for the fetch tool. Download it manually from gbu.ac.in and compare it with v1. If the numbers differ, the newer revision wins.
- **2022 fee structure.** Outdated, so do not use it.

---

## How to use this file
1. Add a record to `knowledge_base.json` only for a fact in the Verified section.
2. Copy the source link and the "Checked on" date into that record.
3. For anything in the Not verified section, either confirm it first or leave it out.
