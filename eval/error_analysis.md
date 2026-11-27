# Error analysis (Phase 5)

Setup: MiniLM + intent classifier, threshold 0.67 (the app's current settings), **test** split.
Numbers come from `eval/results/results.md`. Written by hand, so update it when the results change.

The bot failed on 25 of 69 test questions:

- 19 answerable questions got the fallback.
- 5 answerable questions got the wrong record.
- 1 question that should be refused got an answer.

## Failures by cause

### 1. Right record found, but the score was below the threshold (10)

The correct record was ranked first, but the similarity was under 0.67, so the bot refused.
A lower threshold would answer these, but it would also give many more wrong answers (see the sweep table).

| Question | Score |
|---|---|
| what will i pay per semester for computer engineering | 0.642 |
| सीएसई बीटेक का शुल्क कितना है | 0.581 |
| लाइब्रेरी फीस हर सेमेस्टर कितनी लगती है | 0.554 |
| ladkiyon ke liye kitne hostal hai (typo) | 0.309 |
| लड़कियों के कितने हॉस्टल हैं | 0.536 |
| छात्राओं को दूसरे साल से सिंगल रूम मिलता है क्या | 0.638 |
| abhi admission le sakte hai kya gbu me | 0.644 |
| इस साल दाखिला मिल सकता है क्या | 0.497 |
| What is the full address of Gautam Buddha University? | 0.574 |
| sc/st scholarshp kitni hai (typo) | 0.655 |

### 2. Paraphrase gap: no knowledge base question is worded like this (10)

A different record was ranked first, because none of the correct record's questions are close in wording.

| Question | Got | Note |
|---|---|---|
| cse btech krne me har sem kitna paisa lagta hai | fallback (top: contact_address) | No "fee" word; "paisa" is not in any paraphrase |
| काउंसलिंग फीस कितनी है | fallback (top: hostel_fees_2026) | No Hindi paraphrase for the registration fee |
| library aur sports ke liye alag se paise dene padte hai kya | fallback (top: contact_admissions) | Intent said "other" (0.58), just under the refusal level |
| how much are the mess charges per year | fallback (top: fee_library_sports_2026) | "Mess" is in the hostel fee answer but in none of its questions |
| हॉस्टल में कमरा पाने के लिए क्या करना होगा | fallback (top: hostel_room_type_girls) | No Hindi paraphrase for allotment |
| admisson office ka phone no. | fallback (top: contact_hostel) | Typo plus "phone no." |
| इंटरनेशनल अफेयर्स का नंबर बताइए | fallback (top: contact_hostel) | No Hindi paraphrase for international contact |
| विश्वविद्यालय का पिन कोड क्या है | fallback (top: contact_admissions) | The pin code is in the address answer but in none of its questions |
| मेधावी छात्रों के लिए कोई स्कॉलरशिप है क्या | fallback (top: scholarship_sc_st) | "मेधावी" (meritorious) is not in the knowledge base |
| hostel aur mess ka kharcha kitna aata hai saal ka | **contact_hostel (0.782)** | Wrong answer. Same "mess" gap as above |

### 3. Typo (2)

| Question | Got |
|---|---|
| securty fee wapis milti hai ya nahi | **fee_registration_2026 (0.702)** |
| gbu ka adress bata do | **contact_hostel (0.689)** |

### 4. Ambiguous question (2)

| Question | Got | Note |
|---|---|---|
| डबल रूम का सालाना किराया कितना है | **hostel_room_type_girls (0.780)** | Asks for rent, but "double room" is the wording of the room-type record |
| videshi student hu admission ke liye kisse baat karu | **contact_admissions (0.686)** | The admissions office is a partly reasonable answer; international affairs is better |

### 5. Missing data (1)

| Question | Got | Note |
|---|---|---|
| law ki fees kitni hai gbu me | **fee_security_2026 (0.699)** | The knowledge base has no law fee. "... fees kitni hai" matches the generic fee paraphrases |

## Main findings

1. **Hindi is the weakest language.** 11 of the 15 Hindi test questions failed. The knowledge base has only one Devanagari paraphrase in total. The intent classifier is not the problem: it routes Hindi questions to the right topic, often with 0.8 to 0.97 confidence. The gap is in the knowledge base.
2. **Some facts are in the answers but not in the questions.** Examples are mess charges, the pin code, and the full university name, since the knowledge base says only "GBU". Retrieval compares only against questions and paraphrases, never against the answer text.
3. **`contact_hostel` pulls in other questions.** Four failures landed on it: hostel costs, the admissions phone number, the international number, and the address. It has the most Hinglish paraphrases and the only Devanagari one, so short "... ka number" style questions match it. Paraphrases should be balanced across records.
4. **One threshold cannot separate right from wrong well.** On test, a threshold of 0.50 gives 64.4% right answers with a 31.9% wrong-answer rate. A threshold of 0.67 gives 46.7% right with 8.7% wrong. For a fees and admissions bot, a wrong answer is worse than a fallback, so 0.67 was kept.
5. **The bigger model helps Hindi.** mpnet answered 8 of 15 Hindi test questions correctly, against 4 of 15 for MiniLM. Its overall accuracy is higher by only 3 questions, which is within noise, and it is about 2.4 times bigger and slower.

## How sure are these numbers?

Each split has 69 questions, 45 of them answerable. One question equals about 1.4 points of overall accuracy, or 2.2 points of answer accuracy. Differences of 2 or 3 questions, such as with and without intent, or MiniLM against mpnet overall, are within noise.

The dev score at the chosen threshold was 75.4%, and the test score was 63.8%. The tuned value looks better on the data it was tuned on, which is why only test numbers are reported.

## Next steps

These fixes should come from **dev** failures and general patterns. Never copy test questions into the knowledge base, or the test numbers stop being honest.

1. Add 2 or 3 Devanagari paraphrases and 1 or 2 more Hinglish paraphrases to **every** record, keeping the counts balanced.
2. Add paraphrases for facts that appear only in answers (mess charges, pin code, "Gautam Buddha University").
3. Re-run `python -m eval.evaluate`, re-tune the threshold on dev, and update `SIMILARITY_THRESHOLD`.
4. Decide whether mpnet's Hindi gain is worth its size and speed once the knowledge base is fixed.
