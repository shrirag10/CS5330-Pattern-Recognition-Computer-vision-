# Talking Script — Transfer Learning vs. Random Initialization

Natural spoken delivery, 11 slides, roughly 60–75 seconds per slide. Total ~11–13 minutes (comfortably inside the 15-minute cap). Trim the ablation or feature-space slide if you need to go shorter.

---

## Slide 1 — Title
"Hi everyone. My project asks a deceptively simple question about transfer learning. We all know that starting from an ImageNet-pretrained network beats training from scratch. But *why* does it win? Is it the pretrained features themselves, or is it just that freezing layers gives the optimizer fewer parameters to fumble with? I built a controlled experiment on ResNet-18 and the six-class Intel scenes dataset to separate those two explanations."

---

## Slide 2 — The Problem
"Here's the trap. When you take a pretrained network, freeze the early layers, and fine-tune the rest, you've actually changed *two* things at the same time. Factor A is feature quality — those frozen weights carry edges, textures, and shapes learned from a million images. Factor B is trainable capacity — freezing also shrinks how many parameters the optimizer can move. A naive pretrained-versus-scratch comparison confounds these. If pretrained wins, you genuinely can't say which factor earned it. So my goal was a design that pulls A and B apart."

---

## Slide 3 — The Design
"The fix is a third condition. I take ResNet-18 and split it into early layers and late layers, then run three variants. Pretrained-Frozen: ImageNet weights in the frozen part, train the tail. Random-Full: everything random, train everything. And the key one in the middle — Random-Frozen — random weights in the frozen part, but frozen with the *exact same* trainable tail. That middle condition is the control. Comparing Pretrained-Frozen to Random-Frozen isolates feature quality, because capacity is now identical. Comparing Random-Frozen to Random-Full isolates capacity, because both start random. Two clean comparisons, one variable each."

---

## Slide 4 — Method
"Concretely, the frozen part is conv1 through layer3 — about 2.8 million parameters. The trainable tail is layer4 plus the classifier — about 8.4 million. Pretrained-Frozen and Random-Frozen share that trainable count exactly, which is what makes them a fair fight. One technical detail that matters more than it looks: batch-norm. If you freeze a block but let its batch-norm keep updating running statistics, the representation drifts and training falls apart. So in both frozen conditions I force batch-norm into eval mode. In Random-Full nothing is frozen, so batch-norm updates normally — that difference is by design, not a bug."

---

## Slide 5 — Setup
"Everything else is held constant so the comparison stays honest. Same data — six scene classes, a stratified 90/10 train-val split, and 3,000 test images held completely out. Same recipe for all three — plain SGD, identical hyperparameters, early stopping on validation loss. And for rigor, every condition runs across three seeds, I report mean plus-or-minus standard deviation instead of a single lucky run, and the test set is touched exactly once, after model selection."

---

## Slide 6 — Results: Accuracy and Cost
"Here's the headline. On the left: Pretrained-Frozen hits 93.3% test accuracy, Random-Full — trained from scratch — gets 84.7%, and the control, Random-Frozen, lands at 66.5%. The comparison that matters is Pretrained-Frozen versus Random-Frozen: a 26.7-point gap, with *identical* trainable parameters and the identical recipe. The only thing that changed is whether the frozen features were pretrained — so that gap is pretraining itself, cleanly isolated from capacity. On the right, the cost side: pretraining converges in about 8 epochs, Random-Full needs 15 and roughly four times the wall-clock. So capacity buys back accuracy but you pay in compute, and neither route from random reaches the pretrained ceiling under a small-data, early-stopping budget."

---

## Slide 7 — Training Dynamics
"This is why the gap shows up so fast. On the left, validation accuracy per epoch: pretrained-frozen starts near 94% on epoch one and just stays flat — the features are already good, so there's almost nothing to learn. Random-full climbs slowly over 12 to 17 epochs to a lower plateau. And random-frozen is the noisy band at the bottom, bouncing around in the 60s with big validation-loss spikes on the right — that's the optimizer struggling to map arbitrary frozen features onto classes. So pretraining doesn't just win on final accuracy; it's usable almost immediately."

---

## Slide 8 — Feature Space
"To see *why* the pretrained features are so good, I took the 512-dimensional features from the layer just before the classifier and projected them to 2D with PCA. On the left, pretrained-frozen: the six classes fall into clean clusters — and notice the overlaps, glacier-mountain-sea sit together and buildings-street sit together, which are exactly the confusions we'll see in a moment. On the right, random-frozen: everything's piled on top of everything. The classifier on the left is handed separable structure; on the right it's handed mush. That one picture explains the whole accuracy gap."

---

## Slide 9 — Ablation: Freeze Depth
"I froze at layer4 — but was that the right place to cut? So I swept it. Starting from just a linear probe — training only the final classifier on top of fully frozen pretrained features — I already get 91.3%. Unfreezing layer4 brings it to 92.9%, about 1.6 points. And then nothing: unfreezing layer3, layer2, or the entire network all land at 92.8%, with three-and-a-half times the trainable parameters and zero benefit. So the elbow is exactly at layer4, which validates the boundary I chose, and the bigger message is that almost all of pretraining's value is available with barely any fine-tuning. This sweep used a 40% data subset to keep it tractable on CPU, so the absolute numbers sit a touch below the headline."

---

## Slide 10 — How Far Up Does the First Layer Reach? (Gabor)
"This one came out of a question from Professor Maxwell. Instead of learning the first layer, I replaced it with a fixed bank of Gabor filters — classic oriented edge detectors — froze it, and then asked how far up the network I have to let things re-adapt to cope with that swapped front end. The red curve is the normal learned first layer, the blue is Gabor. With Gabor, a read-out only gets 81%, about ten points below the learned first layer, because the fixed filters produce activations the pretrained upper layers weren't built for. But unfreezing just the very next stage, layer1, recovers five of those points, and after that the curve is flat — layer2, layer3, layer4 add almost nothing. So the disturbance from changing the first layer is absorbed within about one stage; its influence doesn't reach far up. The twist is that a roughly seven-point gap to the learned first layer never closes, even when everything above conv1 is trainable. So the learned first-layer filters carry something a fixed Gabor bank plus full retraining simply can't reconstruct."

---

## Slide 11 — Error Analysis
"It's also worth checking *what* the models get wrong. This is the Pretrained-Frozen confusion matrix. The two real error clusters are Glacier confused with Mountain — 71 cases — and Buildings with Street. And those make sense: glaciers and mountains share snow, ridges, and rock texture; streets are full of buildings and vice-versa. The important part is that the *same* confusions show up in all three conditions — just much larger for random init. That tells me the errors track genuine visual overlap in the data, not a broken training setup."

---

## Slide 12 — Conclusion
"So, three takeaways. One: pretraining's advantage is the feature quality, not the parameter count — capacity-matched, it's worth 26.7 points. Two: capacity helps but can't close the gap — from scratch you reach 84.7%, still about 9 points short, at four times the compute. Three: for small data and tight budgets, transfer learning wins outright, on both accuracy and speed. And the reason I can say that with confidence rather than hand-waving is the random-frozen control. That's what turned a confounded comparison into a clean one. Thank you — happy to take questions."

---

### Q&A prep (anticipated)
- **"Is a frozen *random* feature extractor even a meaningful baseline?"** Yes — random features are a known, studied baseline; they still form a fixed nonlinear projection the tail can partly exploit. It's the honest capacity-matched control for the pretrained case.
- **"Why freeze at layer3 / train only layer4?"** Early layers learn generic edge/texture filters that transfer; the last stage is task-specific. This split follows the standard transferability findings and keeps the trainable count identical across the two frozen conditions.
- **"Std on convergence?"** Epochs vary more across seeds than final accuracy does (e.g. Random-Full 15 ± 2.2 epochs), which is why I report the spread, not just the mean.
- **"How does 93.3% compare to published baselines?"** Community baselines on Intel Scenes with pretrained backbones sit in the low-to-mid 90s, so frozen-transfer here recovers essentially the full fine-tuning accuracy.
