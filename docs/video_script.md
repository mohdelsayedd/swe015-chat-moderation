# Demo video script (max 2:00, narrated)

The brief: only the first 2:00 are watched. The video must show the app **running live with new
input**, at least two inputs including one difficult case, plus the **architecture diagram** and
the **results table**, with voice narration.

**Before recording:** charger plugged in · Settings → System → Power & battery → Power mode:
**Best performance** · close other heavy apps · start the app from the project folder
(`python app.py`) and send one throw-away message so it's warm · open the deck in another tab
of the same browser (slide 4 = architecture, slide 8 = results) · clear the chat
("Clear chat and log") right before you start.

**Record:** Windows Snipping Tool → **Record** (or Win + Shift + R) → select the browser window →
make sure the **microphone** is on → Start. Rehearse once; say what the app *actually* shows.

| Time | Screen | Say (roughly) |
|---|---|---|
| 0:00–0:15 | Deck, slide 1 (cover) | “A gaming platform must flag offensive chat messages in real time on a CPU-only server. I built a chat-moderation app and compared five models, from a rule baseline to a quantized DistilBERT.” |
| 0:15–0:30 | App tab. Model = DistilBERT INT8 (default). Type: `gg well played everyone, see you next match` | “This is the app running live on my laptop CPU with the quantized model. A normal message is labelled clean with its confidence, and the response time is logged on the right.” |
| 0:30–0:45 | Type: `you are a useless idiot, uninstall the game` | “An insult is flagged as offensive and masked in the chat. Latency stays around [X] milliseconds, under my 50 ms budget.” (read the real number) |
| 0:45–1:15 | **Difficult case.** Type: `stuuupid n00b`. Then switch the dropdown to *Rule baseline* and send it again; point at the “All models on the last message” table | “A hard case: stretched letters. [Say what DistilBERT did, e.g. ‘The Transformer calls it clean’.] The rule baseline normalizes the letters and matches ‘stupid’ by edit distance, which is shown here. The side table shows all five models on the same message.” |
| 1:15–1:30 | Deck, slide 4 (architecture) | “Architecture: the Gradio app on the CPU; a regex normalizer feeds the three classical models, a WordPiece tokenizer feeds DistilBERT in FP32 and the default INT8 version, quantized at start-up from the weights on the Hugging Face Hub.” |
| 1:30–1:45 | Deck, slide 8 (results) | “On the same 860 test tweets, quantized DistilBERT has the best macro-F1, 0.81, at half the size of FP32, and with one CPU thread it averages 48.8 ms, within the budget.” |
| 1:45–2:00 | Deck, slide 10 or 11 | “Its weakness is stretched letters, which the simple rule catches. Next I would normalize text before DistilBERT. Thanks for watching.” |

**After recording:** check the length is ≤ 2:00 and the voice is audible → upload to **YouTube
(Unlisted)** or **Google Drive (Anyone with the link)** → open the link in a **private window**
(Edge: Ctrl + Shift + N) to check it plays without logging in → put the link on slide 1 and in the
submission form.
