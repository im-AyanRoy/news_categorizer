"""
News Article Topic Categorizer
------------------------------
A desktop GUI (Tkinter) that classifies a news article into a topic using
TF-IDF features + a Multinomial Naive Bayes classifier (scikit-learn).

Setup:   pip install scikit-learn
Run:     python news_categorizer.py

Features
  * Paste or load (.txt) an article and click "Categorize"
  * Shows the predicted topic, confidence, and probability for every topic
  * Ships with a small built-in training set (6 topics)
  * "Train on CSV" retrains on your own dataset (columns: text,category)
"""
import csv
import random
import tkinter as tk
from collections import Counter
from tkinter import ttk, filedialog, messagebox

from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

DATA = {
    "Politics": [
        "The parliament passed the new election reform bill after a heated debate between the ruling party and the opposition.",
        "The prime minister met foreign ministers to discuss trade agreements and diplomatic relations ahead of the summit.",
        "Senators voted on the budget proposal as the government faces pressure over taxes and public spending.",
        "Opposition leaders demanded a no-confidence vote, accusing the cabinet of mishandling the coalition.",
        "Voters head to the polls on Sunday as candidates make final campaign speeches across the country.",
        "The president signed an executive order on immigration policy, drawing criticism from lawmakers.",
        "The supreme court ruled on the constitutional challenge to the state assembly's new law.",
        "Party leaders announced alliance talks before the upcoming general election and seat sharing.",
    ],
    "Sports": [
        "The striker scored twice in the second half as the team won the league match at home.",
        "India beat Australia by six wickets in the final cricket match, with the captain scoring a century.",
        "The tennis champion advanced to the semifinals after a straight sets victory at the grand slam.",
        "The coach praised his players after the club qualified for the championship final and lifted the trophy.",
        "The Olympic sprinter broke the national record in the 100 metres at the athletics meet.",
        "The basketball team clinched the playoff spot with a late three-pointer in overtime.",
        "The goalkeeper saved a penalty in the shootout to send the team into the World Cup quarterfinals.",
        "The transfer window closed as the football club signed a young midfielder on a five-year contract.",
    ],
    "Technology": [
        "The company unveiled a new smartphone with an upgraded processor, better camera and faster software updates.",
        "Researchers released an open-source artificial intelligence model that improves machine learning accuracy.",
        "A major cybersecurity breach exposed user data as hackers exploited a software vulnerability.",
        "The startup launched a cloud computing platform that lets developers deploy applications faster.",
        "Chipmakers are racing to build smaller semiconductors to power data centers and AI servers.",
        "The latest operating system update adds new privacy features, battery improvements and app controls.",
        "Engineers demonstrated a robot that uses computer vision and neural networks to sort packages.",
        "Social media platforms are rolling out new algorithms and tools to detect fake accounts and bots.",
    ],
    "Business": [
        "Stock markets rallied as investors cheered strong quarterly earnings and falling inflation.",
        "The central bank raised interest rates to control inflation, pushing bond yields higher.",
        "The retail giant announced a merger and acquisition deal worth billions of dollars.",
        "Oil prices fell as global demand weakened, hurting energy company shares and exports.",
        "The company reported record revenue and profit growth, and raised its annual forecast for shareholders.",
        "Unemployment dropped to a new low as the economy added jobs in manufacturing and services.",
        "The rupee weakened against the dollar as foreign investors pulled money out of the stock market.",
        "A fast-growing startup raised funding from venture capital firms at a billion-dollar valuation.",
    ],
    "Entertainment": [
        "The blockbuster film opened to record box office collections over the weekend, thrilling fans.",
        "The actor and director announced their next movie, a big-budget epic with a star-studded cast.",
        "The singer released a new album and will begin a world concert tour next spring.",
        "The streaming series won several awards at the ceremony, including best drama and best actress.",
        "Critics praised the film's cinematography, soundtrack and powerful performances at the festival.",
        "The celebrity couple walked the red carpet at the premiere before the movie's release.",
        "The trailer for the highly anticipated sequel crossed fifty million views within a day.",
        "The music festival lineup features pop stars, rock bands and famous DJs from around the world.",
    ],
    "Health": [
        "Doctors recommend regular exercise and a balanced diet to lower the risk of heart disease and diabetes.",
        "The health ministry launched a vaccination drive as hospitals report a rise in flu cases.",
        "A new study found that the drug reduces symptoms of cancer in patients during clinical trials.",
        "Experts warn that poor sleep and stress can harm mental health and weaken the immune system.",
        "The hospital opened a new unit for treating patients with kidney disease and surgery recovery.",
        "Nutritionists say that eating more fruit, vegetables and fibre improves digestion and overall wellness.",
        "The World Health Organization issued guidelines to prevent the spread of infections and viruses.",
        "Researchers developed a new treatment for asthma that doctors hope will reduce hospital visits.",
    ],
}


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("News Article Topic Categorizer")
        self.geometry("940x580")
        self.minsize(800, 500)
        self.model = None
        self.bars = {}
        self._build_ui()
        texts, labels = [], []
        for cat, items in DATA.items():
            texts += items
            labels += [cat] * len(items)
        self.train(texts, labels, "built-in sample data")

    # ---------------- UI ----------------
    def _build_ui(self):
        self.status = ttk.Label(self, text="", anchor="w", relief="sunken", padding=4)
        self.status.pack(side="bottom", fill="x")

        left = ttk.Frame(self, padding=10)
        left.pack(side="left", fill="both", expand=True)
        ttk.Label(left, text="News article", font=("Segoe UI", 12, "bold")).pack(anchor="w")

        box = ttk.Frame(left)
        box.pack(fill="both", expand=True, pady=6)
        self.text = tk.Text(box, wrap="word", font=("Segoe UI", 11), undo=True, padx=6, pady=6)
        sb = ttk.Scrollbar(box, command=self.text.yview)
        self.text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)
        self.text.bind("<Control-Return>", lambda e: self.classify())

        row = ttk.Frame(left)
        row.pack(fill="x")
        for label, cmd in [
            ("Categorize", self.classify),
            ("Clear", self.clear),
            ("Load article (.txt)", self.load_article),
            ("Sample", self.sample),
            ("Train on CSV", self.load_csv),
        ]:
            ttk.Button(row, text=label, command=cmd).pack(side="left", padx=(0, 6))

        right = ttk.Frame(self, padding=10)
        right.pack(side="right", fill="y")
        ttk.Label(right, text="Predicted topic", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.result = ttk.Label(right, text="—", font=("Segoe UI", 24, "bold"), foreground="#1a56db")
        self.result.pack(anchor="w", pady=(4, 0))
        self.conf = ttk.Label(right, text="Paste an article and click Categorize")
        self.conf.pack(anchor="w")
        ttk.Label(right, text="Topic probabilities", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(16, 4))
        self.bars_frame = ttk.Frame(right)
        self.bars_frame.pack(fill="x")

    def build_bars(self, classes):
        for w in self.bars_frame.winfo_children():
            w.destroy()
        self.bars = {}
        for c in classes:
            r = ttk.Frame(self.bars_frame)
            r.pack(fill="x", pady=2)
            ttk.Label(r, text=c, width=14).pack(side="left")
            pb = ttk.Progressbar(r, length=150, maximum=100)
            pb.pack(side="left", padx=4)
            pct = ttk.Label(r, text="0.0%", width=7)
            pct.pack(side="left")
            self.bars[c] = (pb, pct)

    # ---------------- Model ----------------
    def train(self, texts, labels, source):
        self.model = make_pipeline(
            TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True),
            MultinomialNB(alpha=0.3),
        )
        self.model.fit(texts, labels)
        self.build_bars(list(self.model.classes_))
        acc = ""
        k = min(5, min(Counter(labels).values()))
        if k >= 2:
            try:
                cv = StratifiedKFold(k, shuffle=True, random_state=1)
                score = cross_val_score(clone(self.model), texts, labels, cv=cv).mean()
                acc = f" | cross-validated accuracy: {score:.0%}"
            except Exception:
                pass
        self.status.config(
            text=f"Trained on {len(texts)} articles, {len(set(labels))} topics ({source}){acc}"
        )
        self.result.config(text="—")
        self.conf.config(text="Paste an article and click Categorize")

    def classify(self):
        article = self.text.get("1.0", "end").strip()
        if len(article.split()) < 5:
            messagebox.showwarning("Too short", "Please enter at least a few words of the article.")
            return
        probs = self.model.predict_proba([article])[0]
        classes = self.model.classes_
        best = probs.argmax()
        self.result.config(text=classes[best])
        note = " (low confidence)" if probs[best] < 0.4 else ""
        self.conf.config(text=f"Confidence: {probs[best]:.1%}{note}")
        for c, p in zip(classes, probs):
            pb, pct = self.bars[c]
            pb["value"] = p * 100
            pct.config(text=f"{p:.1%}")

    # ---------------- Actions ----------------
    def clear(self):
        self.text.delete("1.0", "end")
        self.result.config(text="—")
        self.conf.config(text="Paste an article and click Categorize")
        for pb, pct in self.bars.values():
            pb["value"] = 0
            pct.config(text="0.0%")

    def sample(self):
        cat = random.choice(list(DATA))
        self.text.delete("1.0", "end")
        self.text.insert("1.0", random.choice(DATA[cat]))

    def load_article(self):
        path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if path:
            with open(path, encoding="utf-8", errors="ignore") as f:
                self.text.delete("1.0", "end")
                self.text.insert("1.0", f.read())

    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        try:
            texts, labels = [], []
            with open(path, newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                cols = {c.lower().strip(): c for c in (reader.fieldnames or [])}
                if "text" not in cols or "category" not in cols:
                    raise ValueError("CSV must have 'text' and 'category' columns.")
                for row in reader:
                    t, c = (row[cols["text"]] or "").strip(), (row[cols["category"]] or "").strip()
                    if t and c:
                        texts.append(t)
                        labels.append(c)
            if len(set(labels)) < 2:
                raise ValueError("Need at least 2 different categories.")
            self.train(texts, labels, path.split("/")[-1].split("\\")[-1])
        except Exception as e:
            messagebox.showerror("Could not train", str(e))


if __name__ == "__main__":
    App().mainloop()
