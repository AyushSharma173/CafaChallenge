# CAFA 6 Protein Function Prediction: Comprehensive Research Report

---

## Table of Contents

1. [Introduction: What Is This Competition About?](#1-introduction)
2. [Biology Fundamentals: Proteins From First Principles](#2-biology-fundamentals)
3. [Gene Ontology (GO): The Language of Protein Function](#3-gene-ontology)
4. [The CAFA Challenge Series: History and Evolution](#4-cafa-history)
5. [CAFA 6 Competition Details](#5-cafa-6-details)
6. [Evaluation Metrics: How Predictions Are Scored](#6-evaluation-metrics)
7. [Data and Submission Format](#7-data-and-submission)
8. [Baseline Methods: Homology and Sequence Similarity](#8-baseline-methods)
9. [Protein Language Models (PLMs)](#9-protein-language-models)
10. [Deep Learning Approaches for Function Prediction](#10-deep-learning-approaches)
11. [Structure-Based Methods and AlphaFold](#11-structure-based-methods)
12. [Network and Interaction-Based Methods](#12-network-methods)
13. [Text Mining and Literature-Based Methods](#13-text-mining)
14. [Hierarchical Multi-Label Classification](#14-hierarchical-classification)
15. [CAFA 5 Winning Solutions (Detailed Analysis)](#15-cafa-5-solutions)
16. [Key Research Labs, People, and Startups](#16-key-players)
17. [Current Frontier and Open Problems](#17-frontier)
18. [Recommended Strategy for CAFA 6](#18-recommended-strategy)
19. [Key References](#19-references)

---

## 1. Introduction: What Is This Competition About? <a name="1-introduction"></a>

The **CAFA 6 Protein Function Prediction** competition (hosted on Kaggle, $50,000 prize pool) asks participants to predict what biological functions proteins perform, based primarily on their amino acid sequences. Specifically, you must predict **Gene Ontology (GO) terms** — standardized labels that describe protein functions — for a set of target proteins.

This is not a typical Kaggle competition. It is a **prospective (future-data) competition**: many proteins in the test set do not currently have any known functions. During a curation phase after the submission deadline, researchers around the world will experimentally discover functions for some of these proteins. Your predictions are evaluated against these *future discoveries*. This means:

- You cannot look up the answers — they don't exist yet
- Your model must genuinely generalize, not just memorize known annotations
- There will be distribution shift between the leaderboard evaluation and the final evaluation

**Why it matters**: Understanding protein function is fundamental to biology and medicine. Proteins are the molecular machines that perform nearly every task in living cells — from catalyzing chemical reactions to transporting molecules to providing structural support. Knowing what a protein does is the bridge between genomic sequence data and medical applications like drug development, disease understanding, and precision medicine. However, only a small fraction of known protein sequences have experimentally validated functional annotations. Computational prediction is the only scalable approach to close this gap.

---

## 2. Biology Fundamentals: Proteins From First Principles <a name="2-biology-fundamentals"></a>

### 2.1 What Is a Protein?

A **protein** is a large biological molecule (macromolecule) made up of smaller building blocks called **amino acids**. Think of it like a sentence made up of letters — except the "alphabet" has exactly **20 letters** (the 20 standard amino acids), and a typical protein "sentence" is 100–1,000 characters long (though some are much longer).

The 20 amino acids are:

| 1-Letter Code | 3-Letter Code | Name | Key Property |
|---|---|---|---|
| A | Ala | Alanine | Small, hydrophobic |
| R | Arg | Arginine | Positively charged |
| N | Asn | Asparagine | Polar |
| D | Asp | Aspartate | Negatively charged |
| C | Cys | Cysteine | Can form disulfide bonds |
| E | Glu | Glutamate | Negatively charged |
| Q | Gln | Glutamine | Polar |
| G | Gly | Glycine | Smallest, flexible |
| H | His | Histidine | Positively charged (pH-dependent) |
| I | Ile | Isoleucine | Hydrophobic |
| L | Leu | Leucine | Hydrophobic |
| K | Lys | Lysine | Positively charged |
| M | Met | Methionine | Hydrophobic, start codon |
| F | Phe | Phenylalanine | Aromatic, hydrophobic |
| P | Pro | Proline | Rigid, introduces kinks |
| S | Ser | Serine | Polar, phosphorylation site |
| T | Thr | Threonine | Polar, phosphorylation site |
| W | Trp | Tryptophan | Largest, aromatic |
| Y | Tyr | Tyrosine | Aromatic, phosphorylation site |
| V | Val | Valine | Hydrophobic |

### 2.2 From Sequence to Structure to Function

The **central dogma** of molecular biology: DNA → RNA → Protein. The DNA encodes the amino acid sequence of a protein. This sequence is the protein's **primary structure**.

Once synthesized, the amino acid chain **folds** into a specific 3D shape:
- **Primary structure**: The linear sequence of amino acids (e.g., MKWVTFISLLLLFSS...)
- **Secondary structure**: Local folding patterns — alpha-helices (spirals) and beta-sheets (zigzag sheets)
- **Tertiary structure**: The complete 3D shape of a single protein chain
- **Quaternary structure**: How multiple protein chains assemble together

**The key insight**: A protein's 3D structure determines its function. Two proteins with similar structures often have similar functions, even if their sequences have diverged over evolution. However, the sequence determines the structure (this is the protein folding problem that AlphaFold famously solved), so in principle, the sequence contains all the information needed to predict function.

### 2.3 What Does "Function" Mean?

Protein function is multi-faceted. A single protein can:

- **Catalyze a specific chemical reaction** (enzymatic function) — e.g., lactase breaks down lactose
- **Bind to specific molecules** (molecular interaction) — e.g., hemoglobin binds oxygen
- **Participate in biological processes** — e.g., p53 is involved in cell cycle regulation and apoptosis
- **Localize to specific cellular compartments** — e.g., histones are found in the nucleus

A single protein often has **multiple functions** across all these categories. For example, the tumor suppressor protein p53 (UniProt ID: P04637):
- Binds DNA (molecular function)
- Acts as a transcription factor (molecular function)
- Participates in cell cycle arrest (biological process)
- Participates in apoptosis (biological process)
- Localizes to the nucleus (cellular component)
- Localizes to the cytoplasm (cellular component)

This is why protein function prediction is a **multi-label classification** problem — each protein can have many labels simultaneously.

### 2.4 How Do We Know What Proteins Do?

Experimental methods to determine protein function include:
- **Biochemical assays**: Test if a protein catalyzes a specific reaction
- **Gene knockout/knockdown**: Remove or silence the gene encoding the protein and observe the effect
- **Crystallography/Cryo-EM**: Determine 3D structure and infer function from shape
- **Mass spectrometry**: Identify protein-protein interactions
- **Fluorescence microscopy**: Determine where in the cell a protein localizes

These experiments are **slow, expensive, and incomplete**. As of 2025, only ~0.5% of known protein sequences have experimentally validated functional annotations. The rest have either:
- **Electronic annotations**: Computationally inferred (less reliable)
- **No annotations at all**: Function completely unknown

This massive gap between sequence data and functional knowledge is exactly what CAFA aims to address.

### 2.5 Homology: The Evolutionary Basis of Function Prediction

**Homologous proteins** are proteins that share a common evolutionary ancestor. Because evolution is conservative (mutations that break essential functions are selected against), proteins that share similar sequences often share similar functions.

Key concepts:
- **Sequence identity**: The percentage of positions where two aligned sequences have the same amino acid. >30% identity typically indicates similar function; >50% strongly indicates similar function.
- **BLAST (Basic Local Alignment Search Tool)**: The most widely used tool for finding homologous proteins. It searches a database for sequences similar to your query.
- **Homology transfer**: If protein A has known function X, and protein B is homologous to A, predict that B also has function X.

This simple approach is the most important **baseline** in protein function prediction, and remains competitive even against sophisticated deep learning methods — especially for Molecular Function prediction. However, it fails for:
- **Orphan proteins**: Proteins with no detectable homologs
- **Moonlighting proteins**: Proteins that have evolved new functions not shared by homologs
- **Convergent evolution**: Unrelated proteins that evolved the same function independently

---

## 3. Gene Ontology (GO): The Language of Protein Function <a name="3-gene-ontology"></a>

### 3.1 What Is Gene Ontology?

The **Gene Ontology (GO)** is a standardized vocabulary for describing protein (and gene) function. It was created in 1998 to solve the problem of inconsistent terminology across biology — different labs and organisms used different words for the same concepts.

GO provides:
1. A **controlled vocabulary** of ~45,000 terms describing biological concepts
2. A **hierarchical structure** (technically a Directed Acyclic Graph / DAG) organizing these terms from general to specific
3. **Annotations**: Links between proteins and GO terms, with evidence codes indicating how the annotation was determined

### 3.2 The Three Sub-Ontologies

GO is divided into three independent hierarchies (sub-ontologies), each describing a different aspect of function:

#### Molecular Function (MF)
**What the protein does at the molecular level** — its biochemical activity.

Examples (from general to specific):
```
GO:0003674 molecular_function (root)
  └── GO:0005488 binding
        └── GO:0003677 DNA binding
              └── GO:0043565 sequence-specific DNA binding
                    └── GO:0001228 DNA-binding transcription activator activity
  └── GO:0003824 catalytic activity
        └── GO:0016301 kinase activity
              └── GO:0004672 protein kinase activity
                    └── GO:0004674 protein serine/threonine kinase activity
```

**Key insight for prediction**: MF terms tend to be more "sequence-predictable" because molecular function is often determined by specific domains or motifs in the sequence. BLAST-based methods perform relatively well here.

#### Biological Process (BP)
**Which larger biological programs the protein participates in** — the pathways and cellular activities.

Examples:
```
GO:0008150 biological_process (root)
  └── GO:0009987 cellular process
        └── GO:0006950 response to stress
              └── GO:0006974 cellular response to DNA damage stimulus
                    └── GO:0006281 DNA repair
                          └── GO:0000724 double-strand break repair via homologous recombination
  └── GO:0032502 developmental process
        └── GO:0007275 multicellular organism development
```

**Key insight for prediction**: BP terms are harder to predict from sequence alone because the same protein can participate in different biological processes in different cell types or organisms. Context-dependent annotations make this the most challenging sub-ontology.

#### Cellular Component (CC)
**Where in the cell the protein is located or active**.

Examples:
```
GO:0005575 cellular_component (root)
  └── GO:0005622 intracellular anatomical structure
        └── GO:0005634 nucleus
              └── GO:0005654 nucleoplasm
  └── GO:0005576 extracellular region
  └── GO:0016020 membrane
        └── GO:0005886 plasma membrane
```

**Key insight for prediction**: CC prediction has been the most stagnant sub-ontology across CAFA editions. Localization often depends on signal peptides and post-translational modifications that are harder to capture.

### 3.3 The DAG Structure and True Path Rule

GO is structured as a **Directed Acyclic Graph (DAG)**, not a simple tree. This means:
- A term can have **multiple parents** (e.g., "mitochondrial membrane" is a child of both "mitochondrion" and "membrane")
- There are multiple types of relationships: **is_a** (subclass), **part_of**, **regulates**, **has_part**

The **True Path Rule** is critical for prediction:
> If a protein is annotated with a specific GO term, it is implicitly annotated with ALL ancestor terms up to the root.

For example, if a protein has the annotation "protein serine/threonine kinase activity" (GO:0004674), it automatically also has:
- protein kinase activity (GO:0004672)
- kinase activity (GO:0016301)
- catalytic activity (GO:0003824)
- molecular_function (GO:0003674) — the root

**Implication for predictions**: Your predicted probabilities must be **consistent with the hierarchy**. A parent term should always have probability >= its children. If you predict a specific term with probability 0.8, all its ancestors must be >= 0.8. The competition will propagate predictions if needed, but intelligent propagation during modeling improves results.

### 3.4 GO Evidence Codes

Not all GO annotations are equal. They come with **evidence codes** indicating how they were determined:

**Experimental evidence** (high reliability):
- `EXP` — Inferred from Experiment
- `IDA` — Inferred from Direct Assay
- `IMP` — Inferred from Mutant Phenotype
- `IGI` — Inferred from Genetic Interaction
- `IPI` — Inferred from Physical Interaction
- `IEP` — Inferred from Expression Pattern

**Computational evidence** (lower reliability):
- `ISS` — Inferred from Sequence/Structural Similarity
- `IEA` — Inferred from Electronic Annotation (automated, not human-reviewed)

**In CAFA**: Training annotations typically include **experimental evidence only**. The test set evaluation is also based on experimentally validated annotations that accumulate during the curation period. However, **electronic annotations (IEA) can be used as features** — this was a major insight from CAFA 5 winning solutions.

### 3.5 Scale of the Problem

In the CAFA 5 training set:
- ~142,246 proteins with at least one experimental GO annotation
- ~31,466 unique GO terms (21,285 BP, 7,224 MF, 2,957 CC)
- Severe **class imbalance**: Some terms annotate thousands of proteins; most terms annotate fewer than 10
- **Incomplete annotations**: A protein lacking a GO term does NOT mean it doesn't have that function — it just hasn't been experimentally confirmed. This is the **open world assumption** and fundamentally changes the learning problem compared to standard multi-label classification.

---

## 4. The CAFA Challenge Series: History and Evolution <a name="4-cafa-history"></a>

The **Critical Assessment of Functional Annotation (CAFA)** is the premier community challenge for protein function prediction, running since 2010. It uses a unique **prospective evaluation** design where predictions are tested against future experimental discoveries, not existing held-out data.

### 4.1 CAFA 1 (2010–2011)

**Paper**: Radivojac et al. "A large-scale evaluation of computational protein function prediction." *Nature Methods* (2013).

- **Participation**: 54 methods from 30 teams
- **Evaluation**: 866 proteins from 11 organisms, evaluated over a 15-month window
- **Metrics**: Maximum F-measure (Fmax), term-centric AUC
- **Baseline**: BLAST achieved Fmax of 0.38 (MF) and 0.26 (BP)
- **Key findings**:
  - Top methods substantially outperformed BLAST
  - MF prediction was easier than BP prediction
  - Methods combining multiple data sources performed better
  - "Considerable need for improvement" remained

### 4.2 CAFA 2 (2013–2014)

**Paper**: Jiang et al. "An expanded evaluation of protein function prediction methods shows an improvement in accuracy." *Genome Biology* (2016).

- **Key finding**: Top CAFA 2 methods significantly outperformed top CAFA 1 methods — the most dramatic improvement across all CAFA editions
- Ensemble methods and domain-specific predictors showed substantial gains
- Introduced improved semantic precision-recall scoring

### 4.3 CAFA 3 (2016–2017)

**Paper**: Zhou et al. "The CAFA challenge reports improved protein function prediction and new functional annotations for hundreds of genes through experimental screens." *Genome Biology* (2019).

- **Benchmark**: 377 proteins (MF), 717 (BP), 548 (CC)
- **Top method**: **GOLabeler** was the standout in MF
- **Experimental validation**: First CAFA to include large-scale experimental screens
  - 240 biofilm-formation genes in *Candida albicans*
  - 532 biofilm + 403 motility genes in *Pseudomonas aeruginosa*
  - 11 long-term memory genes in *Drosophila melanogaster*
  - Over 1,000 new functional annotations validated
- **Key findings**:
  - Improvement CAFA 2 → CAFA 3 was "less dramatic" than CAFA 1 → CAFA 2
  - CC prediction had NOT improved across editions
  - Methods were becoming increasingly similar
  - For MF, "sequence alignment even overtakes machine learning" as most impactful approach
  - Best methods combined ML with multiple data sources

### 4.4 CAFA 4 (2019–2020)

- Expanded experimental reach and new model organisms
- Enhanced phenotype prediction tasks
- Deep learning and protein language models gained prominence
- Integrated sequence, structure, and network data approaches

### 4.5 CAFA 5 (2023–2024, First on Kaggle)

- **Platform**: First CAFA hosted on Kaggle, dramatically increasing participation
- **Scale**: ~1,675–1,987 teams, 2,850 entries
- **Evaluation metric**: Modified weighted F1-score using information accretion (IA)
- **Winner**: GOCurator (Prof. Shanfeng Zhu, Fudan University) — an ensemble combining text mining, PLM embeddings, structural information, and a deep information retrieval system (GORetriever)
- **2nd Place**: ProtBoost (score: 0.58240) — Py-Boost gradient boosting + ProtT5/ESM2 embeddings + GCN stacking
- **Major shift**: Protein language model embeddings became the dominant feature source, replacing hand-crafted sequence features

### 4.6 CAFA 6 (2025–2026, Current Competition)

- **Platform**: Kaggle
- **Prize pool**: $50,000
- **Timeline**: Started October 15, 2025; submission deadline February 2, 2026; final evaluation June 1, 2026
- **Participation**: 9,108 entrants, 2,259 teams, 779 submissions (as of March 2026)
- **New elements**: Includes partial-knowledge protein targets; optional free-text prediction task; text evaluation using LLMs
- **Periodic leaderboard updates**: 2–3 interim updates expected before final evaluation

---

## 5. CAFA 6 Competition Details <a name="5-cafa-6-details"></a>

### 5.1 The Task

Given a set of protein amino acid sequences (the "test superset"), predict which Gene Ontology terms are associated with each protein, along with confidence scores.

Specifically:
- **Input**: Protein sequences in FASTA format
- **Output**: Triples of (protein_ID, GO_term, probability) — tab-separated, no header
- **Probability range**: (0, 1.000] with up to 3 significant figures. Score of 0 is not allowed; simply omit such pairs.
- **Limit**: Maximum 1,500 GO terms per protein (across all three sub-ontologies combined)
- **Ontology**: Predictions must use valid terms from the provided GO version

### 5.2 Data Provided

| File | Description |
|------|-------------|
| `train_sequences.fasta` | Protein sequences for training (FASTA format) |
| `train_terms.tsv` | GO term annotations for training proteins (protein_ID, GO_term, ontology) |
| `train_taxonomy.tsv` | Protein-to-species taxonomic IDs |
| `testsuperset.fasta` | Test protein sequences to make predictions on |
| GO ontology (OBO format) | Gene Ontology hierarchy |
| IA weights file | Information accretion weights for each GO term |

### 5.3 Three Target Categories

CAFA 6 evaluates three types of proteins:

1. **No-knowledge targets**: Proteins that had NO experimental annotations in a particular sub-ontology before the submission deadline, and accumulate annotations during the curation period
2. **Limited-knowledge targets**: Proteins that had experimental annotations in some but not all three sub-ontologies, and gain new annotations
3. **Partial-knowledge targets** (new in CAFA 6): Proteins that already had experimental annotations in all three sub-ontologies but accumulate additional annotations

The final score is the arithmetic mean of the F-measures across these three knowledge categories, each of which is itself the arithmetic mean of the three sub-ontology F-measures.

### 5.4 Leaderboard vs. Final Evaluation

**Critical warning**: The leaderboard uses a small, curated set of proteins provided by UniProtKB — NOT available in public databases. These proteins will NOT be in the final test set. The final evaluation uses proteins that accumulate real experimental annotations between February 2026 and June 2026.

**Implication**: Overfitting to the leaderboard is dangerous. Focus on generalization. Some distribution shift between leaderboard and final evaluation is expected.

### 5.5 Optional Free-Text Prediction

New in CAFA 6, participants can optionally submit textual descriptions of protein function:
- Up to 5 lines of text per protein
- Each line has a confidence score
- Maximum 3,000 characters per protein
- Will be evaluated using LLMs (Phase 1) and human experts (Phase 2)
- Does not affect GO term leaderboard score or prize eligibility
- Evaluated 9–12 months after submission deadline

---

## 6. Evaluation Metrics: How Predictions Are Scored <a name="6-evaluation-metrics"></a>

### 6.1 Why Standard Metrics Don't Work

Standard metrics like accuracy, AUC, or basic F1-score are inadequate for GO term prediction because:
1. **Hierarchical structure**: Predicting "catalytic activity" (very general) should be worth less than predicting "protein serine/threonine kinase activity" (very specific)
2. **Extreme class imbalance**: The root term applies to every protein; rare specific terms apply to only a handful
3. **Open world**: Missing annotations are not negative examples

### 6.2 Information Accretion (IA)

To address the hierarchy problem, CAFA uses **Information Accretion (IA)** weights, developed by Clark & Radivojac (2013):

```
ia(f) = -log₂(Pr(f | Pa(f)))
```

Where:
- `f` is a GO term
- `Pa(f)` is the set of parent terms of `f`
- `Pr(f | Pa(f))` is the conditional probability of a protein having term `f` given that it has all parent terms

**Intuition**: IA measures how much NEW information a GO term adds beyond what its parents already tell you.
- **Root terms** (e.g., "biological_process"): IA ≈ 0 (every protein has this — no information)
- **Common terms** (e.g., "binding"): Low IA (many proteins bind things)
- **Rare specific terms** (e.g., "UDP-glucose 4-epimerase activity"): High IA (very informative)

The competition provides pre-computed IA weights for all GO terms.

### 6.3 Weighted Precision and Recall

For each protein `i` and threshold `τ`:

**Weighted Precision**:
```
pr_i(τ) = Σ_{f ∈ P_i(τ) ∩ T_i} ia(f) / Σ_{f ∈ P_i(τ)} ia(f)
```

**Weighted Recall**:
```
rc_i(τ) = Σ_{f ∈ P_i(τ) ∩ T_i} ia(f) / Σ_{f ∈ T_i} ia(f)
```

Where:
- `P_i(τ)` = predicted GO terms for protein `i` at threshold `τ` (all terms with score ≥ τ)
- `T_i` = true GO terms for protein `i`
- `ia(f)` = information accretion weight for term `f`

### 6.4 Maximum F-measure (Fmax)

The F-measure at threshold `τ` is the harmonic mean of average precision and recall across all proteins:

```
F(τ) = 2 × avg_precision(τ) × avg_recall(τ) / (avg_precision(τ) + avg_recall(τ))
```

**Fmax** = max over all thresholds `τ` of F(`τ`)

This is computed separately for each sub-ontology (MF, BP, CC), giving three Fmax values. The final score is their arithmetic mean.

### 6.5 CAFA 6 Scoring Pipeline

```
For each knowledge type (no-knowledge, limited-knowledge, partial-knowledge):
    For each sub-ontology (MF, BP, CC):
        Compute weighted Fmax
    Take arithmetic mean of 3 Fmax values → score for this knowledge type
Take arithmetic mean of 3 knowledge-type scores → FINAL SCORE
```

### 6.6 The CAFA-Evaluator Tool

The official evaluation code is available at: https://github.com/BioComputingUP/CAFA-evaluator

This is the authoritative implementation. Use it for local evaluation during development.

---

## 7. Data and Submission Format <a name="7-data-and-submission"></a>

### 7.1 FASTA Format

Training and test sequences are provided in FASTA format:
```
>P9WHI7
MKWVTFISLLLLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQ
YLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRE
...
>P04637
MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQ
WFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYPQG
...
```

### 7.2 Submission Format

Tab-separated, no header, one prediction per line:
```
P9WHI7	GO:0009274	0.931
P9WHI7	GO:0071944	0.540
P9WHI7	GO:0005575	0.324
P04637	GO:1990837	0.23
P04637	GO:0031625	0.989
P04637	GO:0043565	0.64
P04637	GO:0001091	0.49
```

Rules:
- Protein ID must match the FASTA header in the test superset
- GO term must be valid in the provided GO version
- Score must be in (0, 1.000] with up to 3 significant figures
- Maximum 1,500 GO terms per protein (MF + BP + CC combined)
- If predictions are not propagated to the root, they will be automatically propagated (parent gets max of children's scores)
- Proteins not listed are assumed to have all-zero predictions

---

## 8. Baseline Methods: Homology and Sequence Similarity <a name="8-baseline-methods"></a>

### 8.1 BLAST-Based Function Transfer

The simplest and most important baseline:

1. Run BLAST (or faster alternatives like Diamond) on the test protein against a database of annotated proteins
2. Find the top hits (most similar sequences)
3. Transfer GO terms from the hits, weighted by sequence similarity (e.g., E-value or bit score)

**Performance**: BLAST achieved Fmax of ~0.38 (MF) and ~0.26 (BP) in CAFA 1. It remains surprisingly competitive, especially for MF.

**Strengths**:
- Simple and fast
- Leverages billions of years of evolutionary conservation
- Works well when close homologs exist (>30% sequence identity)
- Particularly strong for molecular function, where conserved domains directly determine activity

**Weaknesses**:
- Fails completely for orphan/novel proteins with no detectable homologs
- Performance degrades rapidly below ~30% sequence identity ("twilight zone")
- Cannot predict novel functions not seen in training
- Cannot leverage non-sequence information (expression, interactions, literature)

### 8.2 Diamond

**Diamond** is a faster alternative to BLAST (100–20,000× faster) that produces nearly identical results. For large-scale function prediction, Diamond is preferred. Used in DeepGOPlus as the alignment component.

### 8.3 InterPro and Pfam Domain-Based Transfer

**InterPro** is a database that classifies proteins into families and predicts domains and functional sites. **Pfam** is one of its member databases focused on protein domain families.

**InterPro2GO** is a manually curated mapping from InterPro entries to GO terms:
- ~91–100% accuracy for curated mappings
- The single largest source of automated GO annotations in UniProt
- Running InterProScan on a protein gives you domain-based function predictions "for free"

**In CAFA 5 winners**: Electronic annotations from InterPro/Pfam were used as features and provided substantial performance boosts.

### 8.4 Naive Baseline

The simplest possible baseline: predict the prior frequency of each GO term across all annotated proteins. Every protein gets the same predictions. This is the "naive" baseline that any useful method should beat.

---

## 9. Protein Language Models (PLMs) <a name="9-protein-language-models"></a>

### 9.1 What Are Protein Language Models?

Protein Language Models (PLMs) apply the same principles as large language models (GPT, BERT) but to protein sequences instead of natural language. Just as ChatGPT learns the "grammar" of English from massive text corpora, PLMs learn the "grammar" of proteins — the patterns, motifs, and dependencies in amino acid sequences — from hundreds of millions of protein sequences.

**Key insight**: PLMs learn rich representations (embeddings) of proteins that capture evolutionary, structural, and functional information, WITHOUT being explicitly trained on function labels. These embeddings can then be used as features for downstream tasks like function prediction.

### 9.2 How PLMs Work

**Training objective**: Most PLMs use **masked language modeling** (like BERT). Random amino acids in a sequence are masked, and the model learns to predict them from context.

Example:
```
Input:  M K W V [MASK] F I S L L [MASK] L F S S A
Target: M K W V T     F I S L L L     L F S S A
```

To predict the masked positions correctly, the model must learn:
- Which amino acids are likely at each position
- Long-range dependencies (amino acids far apart in sequence can interact in 3D)
- Evolutionary conservation patterns
- Structural constraints

**Output**: For each amino acid position, the model produces an embedding vector (e.g., 1,280 dimensions for ESM-2 650M). These per-residue embeddings encode the model's understanding of that position in context.

### 9.3 Major PLMs

#### ESM-2 (Meta/FAIR, 2022)

**Paper**: Lin et al. "Language models of protein sequences at the scale of evolution predict structure and function." *Science* (2022).

| Variant | Parameters | Layers | Embedding Dim |
|---------|-----------|--------|---------------|
| ESM-2 8M | 8M | 6 | 320 |
| ESM-2 35M | 35M | 12 | 480 |
| ESM-2 150M | 150M | 30 | 640 |
| ESM-2 650M | 650M | 33 | 1,280 |
| ESM-2 3B | 3B | 36 | 2,560 |
| ESM-2 15B | 15B | 48 | 5,120 |

- Architecture: RoBERTa-based (encoder-only transformer)
- Training data: UniRef50
- **ESM-2 650M** is the most commonly used variant — best performance/cost tradeoff
- Can predict protein structure directly from its internal representations
- GitHub: https://github.com/facebookresearch/esm

**Benchmark performance** (Fmax on CAFA4 dataset):
- ESM-2 3B: BP=0.452, MF=0.620, CC=0.734
- ESM-2 650M: MF=0.670, CC=0.671 (on Human2024 dataset)

#### ESM-1b (Meta/FAIR, 2021)

- 650M parameters, 33 layers, 1,280-dim embeddings
- Predecessor to ESM-2
- Surprisingly, in comprehensive benchmarks, ESM-1b emerged as the **top performer** across most function prediction benchmarks — slightly outperforming even ESM-2 in several tests
- Fmax on CAFA4: BP=0.456, MF=0.626, CC=0.736

#### ESM-3 (EvolutionaryScale, 2025)

**Paper**: Hayes et al. "Simulating 500 million years of evolution with a language model." *Science* (2025).

- 98B parameters (full version)
- **Multimodal**: Jointly reasons over sequence, structure, AND function
- Uses geometric attention for structure
- Not primarily for function prediction, but demonstrates deep understanding of protein biology
- Generated a novel bright fluorescent protein with only 58% identity to known fluorescent proteins

#### ESM Cambrian (ESM-C, EvolutionaryScale, December 2024)

- Latest representation-focused model from EvolutionaryScale
- 300M variant matches ESM-2 650M performance
- 600M variant rivals ESM-2 3B
- Best current option for extracting high-quality embeddings
- Available on HuggingFace

#### ProtT5-XL-U50 (ProtTrans, TU Munich, 2021)

**Paper**: Elnaggar et al. "ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning." *IEEE TPAMI* (2021).

- 1.2B parameters, T5 architecture (encoder-decoder)
- 1,024-dimensional per-residue embeddings
- Trained on BFD (2.1B sequences) + UniRef50
- The **most widely used PLM** in CAFA 5 winning solutions
- Particularly popular because pre-computed embeddings are available on HuggingFace
- HuggingFace: `Rostlab/prot_t5_xl_uniref50`

#### ProstT5 (TU Munich, 2023)

- 1.2B parameters, T5 architecture
- Extends ProtT5 to also process **3Di structural sequences** (from Foldseek)
- Bidirectional sequence ↔ structure translation
- Can incorporate AlphaFold-predicted structural information

#### Ankh (2023)

- Optimized follow-up to ProtTrans
- Base: 450M params; Large: 1.1B params
- Outperforms larger ProtTrans and ESM models by ~4.8% through optimized pretraining
- Used by several CAFA 5 top solutions

### 9.4 How to Use PLM Embeddings for Function Prediction

**Step 1: Extract embeddings**
```python
# Example with ESM-2
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")

inputs = tokenizer("MKWVTFISLLLLFSS", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

per_residue_embeddings = outputs.last_hidden_state  # shape: (1, seq_len, 1280)
```

**Step 2: Pool to protein-level**
```python
# Mean pooling (most common)
protein_embedding = per_residue_embeddings.mean(dim=1)  # shape: (1, 1280)
```

**Step 3: Use as features for classification**
- Train a classifier (MLP, gradient boosting, etc.) on these embeddings
- Each GO term is typically treated as a separate binary classification target

**Practical considerations**:
- **Sequence length limits**: ESM models have context limits (1024 tokens for ESM-1b); truncate longer sequences
- **GPU memory**: Larger models (3B, 15B) require significant GPU memory; use fp16/bf16
- **Pre-computed embeddings**: For competitions, pre-compute and save embeddings to avoid repeated inference
- **Layer selection**: Mean pooling across layers 20–33 (for ESM-2 650M) sometimes outperforms using only the last layer
- **Fine-tuning vs. frozen**: Fine-tuning with LoRA can improve task-specific performance but risks overfitting on small datasets

### 9.5 Comparative Performance Summary

From comprehensive benchmarks (PMC11790633):

| Model | CAFA4 BP | CAFA4 MF | CAFA4 CC |
|-------|----------|----------|----------|
| ESM-1b | **0.456** | **0.626** | **0.736** |
| ESM-2 3B | 0.452 | 0.620 | 0.734 |
| ESM-2 650M | 0.443 | 0.615 | 0.731 |
| ProtT5 | 0.422 | 0.539 | 0.706 |
| ProstT5 | 0.414 | 0.520 | 0.689 |
| ProtBERT | 0.376 | 0.416 | 0.657 |

**Key takeaway**: ESM-1b and ESM-2 are consistently the strongest PLMs for function prediction. Medium-sized models (650M) perform nearly as well as much larger ones (3B, 15B). ProtT5 is slightly behind but widely used due to availability.

---

## 10. Deep Learning Approaches for Function Prediction <a name="10-deep-learning-approaches"></a>

### 10.1 CNN-Based Methods

#### DeepGOPlus (Kulmanov & Hoehndorf, 2020)

The simplest deep learning baseline for GO prediction:
- **Architecture**: 1D CNN on one-hot encoded sequences (21 × 2000 matrix) + Diamond BLAST scores
- **Training**: Separate binary classifiers for each GO term
- **Performance**: Competitive with more complex methods; processes ~40 sequences/second
- **Key insight**: Combining CNN with homology (Diamond) scores gives best of both worlds — the CNN captures motif patterns while Diamond captures global similarity

### 10.2 Transformer-Based Methods

#### ATGO (2022)
- Uses ESM-1b multilayer embeddings
- Triplet neural network architecture with positive/negative GO term samples
- Strong performance with PLM features

#### TALE (2023)
- Self-attention transformer for joint sequence-label embedding
- **Zero-shot capable**: Can predict GO terms never seen during training by leveraging GO term descriptions
- Important for rare/new GO terms

#### TEMPROT
- Uses ProtBERT-BFD
- Sliding window approach for long sequences (500-token chunks)
- Aggregates predictions across windows

### 10.3 GNN-Based Methods (Graph Neural Networks)

#### PANDA2
- GNN that explicitly models the GO DAG structure
- Uses ESM embeddings as protein features
- GNN propagates information through the ontology hierarchy

#### PFresGO (2024)
- Combines ProtT5 embeddings + Anc2vec GO term embeddings
- Non-feedforward residual attention blocks
- Cross-attention between protein and GO representations
- Integrates GO inter-relationships during prediction

#### DeepGO-SE (Kulmanov & Hoehndorf, 2024)

**Paper**: *Nature Machine Intelligence* (2024)

- **Architecture**: ESM-2 embeddings + ELEmbeddings (geometric ontology embeddings)
- **Key innovation**: Treats function prediction as **approximate semantic entailment** — instead of just predicting binary labels, it checks whether the protein "entails" each GO term in a learned semantic space
- **Neuro-symbolic**: Combines neural network representations with logical ontology axioms
- **Performance**: MF Fmax 0.554; significant improvement over DeepGOPlus
- **Zero-shot capability**: Can predict unseen GO terms through ontology axioms

### 10.4 Gradient Boosting Methods

Surprisingly, traditional gradient boosting often outperforms deep learning for function prediction when combined with PLM embeddings:

#### Py-Boost (used in CAFA 5 2nd place)
- Novel multi-target gradient boosting framework
- ~50× faster than XGBoost on 500+ targets
- Trained on ProtT5/ESM-2 embeddings + taxonomy + electronic annotations
- With proper post-processing, achieved 0.58240 on CAFA 5 leaderboard

**Why gradient boosting works**: PLM embeddings are already rich, high-level representations. Gradient boosting is excellent at learning decision boundaries in structured feature spaces, and handles the extreme class imbalance well through sample weighting.

---

## 11. Structure-Based Methods and AlphaFold <a name="11-structure-based-methods"></a>

### 11.1 Why Structure Helps

Protein function is determined by its 3D structure. Two proteins can have very different sequences but similar structures (and thus similar functions) — this is called **structural homology** and extends function prediction beyond what sequence similarity can capture.

Before AlphaFold, experimental 3D structures were available for only ~180,000 proteins (PDB database). AlphaFold changed this by predicting structures for essentially all known proteins (~200M+).

### 11.2 DeepFRI (Gligorijevic et al., 2021)

**Paper**: *Nature Communications* (2021)

The landmark structure-based function prediction method:
- **Architecture**: 3-layer Graph Convolutional Network (GCN) operating on protein contact maps + LSTM language model features
- **Input**: Contact map (which amino acids are physically close in 3D space) + sequence embeddings
- **Performance**: Fmax = 0.657 for MF-GO
- **Key insight**: Maintains strong performance even with predicted (non-experimental) structures — AlphaFold structures work nearly as well as crystal structures
- **Interpretability**: Uses grad-CAM to highlight which residues contribute most to each function prediction

### 11.3 TransFun (2023)
- Equivariant Graph Neural Network (EGNN)
- Uses AlphaFold2 structures + ESM-1b embeddings
- Combines structural and sequence information

### 11.4 Struct2GO
- GCN on 2D contact maps (10Å threshold) + Node2vec + SeqVec embeddings

### 11.5 Practical Considerations

- **AlphaFold structures** are freely available for most proteins via AlphaFold DB
- **Foldseek** can convert 3D structures to "3Di" sequence alphabets, enabling sequence-like analysis of structure
- **ProstT5** integrates structural information through 3Di sequences
- Structure is most helpful for **Molecular Function** prediction, less so for Biological Process

---

## 12. Network and Interaction-Based Methods <a name="12-network-methods"></a>

### 12.1 Protein-Protein Interaction (PPI) Networks

Proteins don't work in isolation — they form networks of interactions. A protein's interaction partners provide strong clues about its function ("guilt by association").

Key databases:
- **STRING**: Largest PPI database, combining experimental, co-expression, and text-mined interactions
- **BioGRID**: Curated physical and genetic interactions
- **IntAct**: Molecular interaction data from literature

### 12.2 NetGO Series (Zhu Lab, Fudan University)

The NetGO series represents the evolution of network-integrated function prediction:

#### NetGO 1.0 (2019)
- Learning-to-Rank framework building on GOLabeler
- Added massive PPI network information
- 14% AUPR improvement over GOLabeler in BP
- **Key insight**: Network information helps most for Biological Process prediction, where cellular context matters

#### NetGO 2.0
- Added text mining from scientific literature (LR-Text)

#### NetGO 3.0 (2023)
- Replaced sequence features with ESM-1b embeddings (LR-ESM)
- Performance: MF Fmax 0.679, BP 0.378, CC 0.670
- ESM embeddings alone provided 21–31% improvement over traditional sequence features

### 12.3 DeepGraphGO
- Graph CNN + InterProScan features
- Multi-species PPI strategy — trains on interactions across species

### 12.4 deepNF (Deep Network Fusion)
- Multimodal autoencoder processing multiple PPI networks
- Random Walk with Restart (RWR) + Positive PMI matrices

---

## 13. Text Mining and Literature-Based Methods <a name="13-text-mining"></a>

### 13.1 Why Text Mining Matters

Scientific literature contains vast amounts of functional information about proteins. Millions of papers describe experimental findings about protein function, but this knowledge is often not yet captured in structured databases like GO.

### 13.2 GORetriever (CAFA 5 Winner Component)

**Paper**: Zhu et al. (2024), PMC11520413

The key innovation from the CAFA 5 winning team:

**Stage 1 — Candidate Retrieval**:
- Uses MonoT5 (T5-based sequence-to-sequence ranker) to extract informative sentences from literature
- BM25 (traditional information retrieval) to find proteins with similar descriptions
- Aggregates GO terms from similar proteins

**Stage 2 — Reranking**:
- Cross-Encoder based on PubMedBERT
- Computes semantic matching scores between protein text features and GO term definitions
- Separate models for MF, BP, CC

**Performance**: MF 0.659, BP 0.545, CC 0.653 (on text-available proteins)

### 13.3 ProtNLM (Google)

- Used LLMs to automatically annotate 28M+ UniProt proteins
- Generates natural language descriptions of protein function from sequence
- Demonstrates the power of combining sequence understanding with language generation

### 13.4 Practical Implications for CAFA 6

Text mining was a major differentiator in CAFA 5. For CAFA 6:
- UniProt entries contain textual descriptions that can be mined
- PubMed abstracts linked to proteins provide additional context
- TF-IDF features from paper abstracts were used by CAFA 5 4th place solution
- Two-tower architectures matching protein representations with GO term text descriptions are being explored

---

## 14. Hierarchical Multi-Label Classification <a name="14-hierarchical-classification"></a>

### 14.1 The Challenge

Protein function prediction is not standard multi-label classification because:
1. Labels (GO terms) are organized in a hierarchy (DAG)
2. The True Path Rule must be satisfied (parent ≥ child)
3. Labels are incomplete (open world)
4. There are ~45,000 possible labels with extreme class imbalance

### 14.2 Approaches to Hierarchy

#### Flat Classification + Post-Processing
The simplest approach: train independent classifiers for each GO term, then enforce hierarchy constraints:
- **Bottom-up propagation**: For each term, set parent score = max(parent_score, child_score)
- **Top-down propagation**: For each term, set child score = min(child_score, parent_score)
- **Dual propagation** (ProtBoost): Apply both directions and average

#### Conditional Probability Modeling (ProtBoost)
- During training, exclude sample-target pairs where ALL parent GO nodes have zero values from the loss function
- During inference: `P_modified(Node) = P_model(Node) × (1 - ∏(1 - P_modified(Parent)))`
- This naturally enforces that a term's probability depends on its parents

#### GCN Stacking (ProtBoost)
- Build a graph neural network that operates on the GO DAG
- Input: predictions from base models as node features
- The GCN learns to propagate and adjust predictions through the hierarchy
- Features per node: 20 from base models, 1 electronic annotation, 8 trainable embeddings = 29 total

#### Ontology-Aware Neural Models
- **DeepGO-SE**: Uses logical axioms from the ontology during training
- **DeepGOZero**: Enables zero-shot prediction of GO terms not seen in training
- **PFresGO**: Cross-attention between protein and GO term representations

### 14.3 Selecting Which GO Terms to Predict

With ~45,000 possible terms, not all can be effectively predicted:
- **CAFA 5 2nd place**: Selected top 3,000 BP, 1,000 MF, 500 CC terms by frequency
- **Rare terms**: Below a minimum frequency threshold, there isn't enough training data
- **Information content**: Focus on terms with moderate-to-high IA weights

---

## 15. CAFA 5 Winning Solutions (Detailed Analysis) <a name="15-cafa-5-solutions"></a>

### 15.1 First Place: GOCurator (Score: Highest)

**Team**: Prof. Shanfeng Zhu, Fudan University (Wei Liu and others)
**Code**: https://github.com/ZhuLab-Fudan/GORetriever

**Architecture**: Diverse ensemble combining:
1. **PLM embeddings**: ProtT5, ESM-2, Ankh
2. **Text mining**: GORetriever deep information retrieval system
3. **Structural information**: 3D structure-based features
4. **Homology**: Sequence similarity-based predictions

**GORetriever** (key differentiator):
- Two-stage deep information retrieval
- Stage 1: MonoT5 + BM25 for candidate GO term retrieval from literature
- Stage 2: PubMedBERT-based cross-encoder for reranking
- Separate models for MF, BP, CC

**Why it won**: The text mining component (GORetriever) provided significant gains, especially for proteins with published literature that hadn't yet been formally annotated. This exploited the "open world" nature of annotations.

### 15.2 Second Place: ProtBoost (Score: 0.58240)

**Team**: Wang, Zhai, Liu, Huang, Yan
**Code**: https://github.com/btbpanda/CAFA5-
**Paper**: arXiv 2412.04529

**Pipeline** (4 stages):

**Stage 1 — Base Models (Py-Boost)**:
- Py-Boost gradient boosting (~50× faster than XGBoost on many targets)
- Primary features: ProtT5 embeddings (1024-dim)
- Supplementary: ESM-2 embeddings, one-hot taxonomy, electronic GO annotations from UniProtKB
- Targets: 3,000 BP + 1,000 MF + 500 CC terms = 4,500 total
- 5-fold cross-validation

**Stage 2 — Conditional Probability Modeling**:
- Excludes impossible parent-child pairs from loss
- Post-processing: `P(node) = P_model(node) × (1 - ∏(1 - P(parent)))`

**Stage 3 — GCN Stacking**:
- 29 features per GO node: 20 from 5 base models × 4 multiplexed, 1 electronic annotation, 8 trainable embeddings
- Forward/backward/bidirectional graph edges
- Residual connections

**Stage 4 — Dual Propagation Post-Processing**:
- Root-to-terminal propagation (enforces P_min)
- Terminal-to-root propagation (enforces P_max)
- Final score = average of P_min and P_max

**Ablation** (CV scores):
| Component | Score |
|-----------|-------|
| Ridge + ProtT5 (baseline) | 0.47 |
| + Py-Boost | 0.51 |
| + Taxonomy features | 0.53 |
| + Conditional Probability Modeling | 0.55 |
| + Electronic GO annotations | 0.57 |
| + Neural network ensemble | 0.59 |
| + GCN stacking | 0.62 |

**Training time**: Single fold ~2 hours on one GPU; full 5-fold: 1.4–4.5 days

### 15.3 Third Place (Score: 0.57276)

**Team**: Tito

**Key innovations**:
- Used computationally hypothesized (non-experimental, IEA) GO labels as additional training features
- Treated train/validation/test split as a **time series** to account for annotation drift — annotations accumulate over time, so temporal awareness is important

### 15.4 Fourth Place (Score: 0.56245)

**Features**:
- Three PLM embeddings: ProtT5, ESM-2, Ankh
- Species taxonomy binary matrix
- TF-IDF from paper abstracts (text mining)
- UMAP/t-SNE dimension reduction on ProtBERT embeddings to 3D before concatenation
- Interesting finding: "Reduced ProtBERT to 3 dims using UMAP...score improved"

### 15.5 Common Patterns Across Top Solutions

1. **PLM embeddings as primary features** — ProtT5 and ESM-2 were near-universal
2. **Gradient boosting ≥ deep learning** for the final classifier (when features are pre-computed)
3. **Electronic annotations as features** — significant performance boost
4. **Taxonomy/species features** — consistent small improvement
5. **Ontology-aware post-processing** — propagation to enforce True Path Rule
6. **Multiple PLMs are better than one** — ensemble diversity matters
7. **Text/literature features** — major differentiator for 1st place

---

## 16. Key Research Labs, People, and Startups <a name="16-key-players"></a>

### 16.1 Academic Researchers

| Person | Affiliation | Contribution |
|--------|------------|-------------|
| **Predrag Radivojac** | Northeastern University | Co-founder of CAFA, evaluation framework designer, information accretion theory |
| **Iddo Friedberg** | Iowa State University | Co-founder of CAFA, organizer of CAFA 5/6 |
| **Shanfeng Zhu** | Fudan University | GOLabeler, NetGO series, GORetriever, CAFA 5 winner |
| **Robert Hoehndorf** | KAUST | DeepGO, DeepGO-SE, DeepGOPlus, DeepGOZero |
| **Burkhard Rost** | TU Munich | ProtTrans/ProtT5, SeqVec, Ankh, foundational PLM work |
| **Alexander Rives** | Meta FAIR / EvolutionaryScale | ESM model family |
| **John Jumper** | Google DeepMind | AlphaFold (structure prediction enabling function prediction) |
| **Damiano Piovesan** | University of Padova | CAFA-evaluator, CAFA 6 organizer |
| **M. Clara De Paolis Kaluza** | Northeastern University | CAFA 6 organizer |

### 16.2 Key Research Labs

| Lab | Focus | Key Outputs |
|-----|-------|-------------|
| **Meta FAIR** → **EvolutionaryScale** | Protein language models | ESM-1b, ESM-2, ESM-3, ESM-C |
| **Rostlab (TU Munich)** | Protein representation learning | ProtTrans, ProtT5, Ankh, SeqVec |
| **Zhu Lab (Fudan University)** | Protein function prediction | GOLabeler, NetGO 1/2/3, GORetriever |
| **Bio-Ontology Research Group (KAUST)** | Ontology-aware prediction | DeepGO, DeepGO-SE, DeepGOZero |
| **Google DeepMind** | Structure prediction & annotation | AlphaFold, ProtNLM |
| **BioComputing UP (Padova)** | CAFA evaluation | CAFA-evaluator tool |

### 16.3 Startups and Companies

| Company | Funding | Focus | Relevance |
|---------|---------|-------|-----------|
| **EvolutionaryScale** | $142M | Protein language models (ESM-3, ESM-C) | Best protein embeddings; spun out from Meta FAIR |
| **Profluent** | $150M | Programmable biology, protein generation | Related protein AI technology |
| **Cradle** | $103M | AI-driven protein engineering | Uses function prediction in design loop |
| **Generate Biomedicines** | Amgen deal | Generative protein design | Related application domain |
| **Absci** | Public | Generative antibody design | Drug discovery applications |
| **Insilico Medicine** | >$400M | AI-driven drug discovery | Uses protein function understanding |

### 16.4 Related Competitions and Benchmarks

| Competition/Benchmark | Focus | Relationship to CAFA |
|---|---|---|
| **CASP (Critical Assessment of Structure Prediction)** | Protein structure prediction | Structure enables function prediction |
| **TAPE (Tasks Assessing Protein Embeddings)** | Benchmarking protein representations | Tests PLMs on function-related tasks |
| **FLIP** | Fitness landscape prediction | Related protein property prediction |
| **ProteinGym** | Mutation effect prediction | Related sequence-function relationship |
| **ARC Prize** | General AI reasoning | Different domain, similar competition format |

---

## 17. Current Frontier and Open Problems <a name="17-frontier"></a>

### 17.1 What's Working Now (State of the Art, 2025)

1. **PLM embeddings + gradient boosting/ensemble** is the dominant paradigm
2. **Multi-source integration** (sequence + text + interactions + structure + electronic annotations) consistently outperforms any single source
3. **ESM-2 and ProtT5** embeddings are the most reliable feature sources
4. **Hierarchical post-processing** (propagation, conditional probability) provides reliable gains

### 17.2 Open Challenges

1. **Biological Process prediction** remains significantly harder than MF or CC — context-dependent, less sequence-determined
2. **Novel protein prediction** — proteins with no homologs or literature remain very difficult
3. **Rare GO terms** — most terms have too few training examples for reliable prediction
4. **Temporal annotation drift** — the distribution of newly annotated proteins may differ from historical data
5. **Open world evaluation** — even the "ground truth" is incomplete; some correct predictions may be scored as false positives

### 17.3 Emerging Approaches

1. **LLM-based function description** (ProtNLM, free-text predictions) — using language models to generate natural language descriptions of function
2. **Multi-modal foundation models** (ESM-3) — jointly modeling sequence, structure, and function
3. **Two-tower architectures** — encoding proteins and GO terms in a shared embedding space, enabling zero-shot and few-shot prediction
4. **Retrieval-augmented prediction** (GORetriever) — combining neural models with information retrieval from literature
5. **Self-supervised structural features** (ProstT5, 3Di sequences) — incorporating structural information without needing explicit 3D coordinates

### 17.4 What Could Win CAFA 6

Based on CAFA 5 patterns and current trends:

1. **An ensemble** combining multiple PLM embeddings (ESM-2, ProtT5, Ankh, ESM-C)
2. **Electronic annotation features** from UniProtKB
3. **Text mining** from literature (this was the 1st-place differentiator in CAFA 5)
4. **Ontology-aware training and post-processing** (conditional probability, GCN stacking, propagation)
5. **Species taxonomy features**
6. **Gradient boosting** (Py-Boost, LightGBM, CatBoost) as base models, with optional neural network stacking
7. **Careful handling of the temporal aspect** — using annotation timestamps to create realistic train/val splits

---

## 18. Recommended Strategy for CAFA 6 <a name="18-recommended-strategy"></a>

### 18.1 Phase 1: Baseline and Data Understanding

1. **Download and explore all competition data**
2. **Parse the GO ontology** (OBO format) — build the DAG structure in memory
3. **Implement the CAFA-evaluator locally** for validation
4. **Build a BLAST/Diamond baseline** — this is your floor
5. **Compute PLM embeddings** for all train and test proteins (ProtT5, ESM-2 650M)

### 18.2 Phase 2: Core Model

1. **Train gradient boosting models** (Py-Boost or LightGBM) on PLM embeddings for top-k frequent GO terms per sub-ontology
2. **Add features**: taxonomy, electronic annotations, InterPro domains
3. **Implement conditional probability modeling** for hierarchy awareness
4. **Implement dual propagation post-processing**

### 18.3 Phase 3: Advanced Features

1. **Add multiple PLM embeddings** (ESM-1b, ESM-2, ProtT5, Ankh, ESM-C)
2. **Implement text mining** from UniProt descriptions and PubMed abstracts
3. **Add BLAST/Diamond similarity scores** as features
4. **Consider GCN stacking** on the GO DAG

### 18.4 Phase 4: Ensemble and Optimization

1. **Blend multiple model predictions** (gradient boosting, neural networks, homology)
2. **Optimize threshold selection** using CAFA-evaluator on validation set
3. **Ensure all predictions respect the GO hierarchy**
4. **Validate submission format** rigorously

### 18.5 Key Pitfalls to Avoid

- **Don't overfit to the leaderboard** — it's a small, non-representative sample
- **Don't ignore electronic annotations** — they're legal and highly informative features
- **Don't treat absent annotations as negatives** — open world assumption
- **Don't forget propagation** — predictions must be consistent with the GO DAG
- **Don't predict too many terms per protein** — maximum 1,500 across all sub-ontologies
- **Don't use a single PLM** — ensemble diversity across PLMs is important

---

## 19. Key References <a name="19-references"></a>

### Competition and CAFA

1. Radivojac P et al. "A large-scale evaluation of computational protein function prediction." *Nature Methods* (2013) 10(3):221-227.
2. Jiang Y et al. "An expanded evaluation of protein function prediction methods shows an improvement in accuracy." *Genome Biology* (2016) 17(1):184.
3. Zhou N et al. "The CAFA challenge reports improved protein function prediction and new functional annotations." *Genome Biology* (2019) 20(1):244.
4. Clark WT, Radivojac P. "Information-theoretic evaluation of predicted ontological annotations." *Bioinformatics* (2013) 29(13):i53-i61.

### Protein Language Models

5. Lin Z et al. "Language models of protein sequences at the scale of evolution predict structure and function." *Science* (2022).
6. Elnaggar A et al. "ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning." *IEEE TPAMI* (2021).
7. Hayes T et al. "Simulating 500 million years of evolution with a language model." *Science* (2025).

### Function Prediction Methods

8. You R et al. "GOLabeler: improving sequence-based large-scale protein function prediction by learning to rank." *Bioinformatics* (2018).
9. Wang Y et al. "NetGO 3.0: Protein Language Model Improves Large-scale Functional Annotations." *Genomics, Proteomics & Bioinformatics* (2023).
10. Kulmanov M, Hoehndorf R. "DeepGOPlus: improved protein function prediction from sequence." *Bioinformatics* (2020).
11. Kulmanov M, Hoehndorf R. "Protein function prediction as approximate semantic entailment." *Nature Machine Intelligence* (2024).
12. Gligorijevic V et al. "Structure-based protein function prediction using graph convolutional networks." *Nature Communications* (2021).

### CAFA 5 Solutions

13. Wang S et al. "ProtBoost: protein function prediction with Py-Boost and Graph Neural Networks." *arXiv* 2412.04529 (2024).
14. Zhu S et al. "GORetriever: reranking protein-description-based GO candidates by literature-driven deep information retrieval." *PMC* (2024).

### Evaluation and Background

15. CAFA-evaluator: https://github.com/BioComputingUP/CAFA-evaluator
16. Gene Ontology: https://geneontology.org/
17. ESM models: https://github.com/facebookresearch/esm
18. ProtTrans: https://github.com/agemagician/ProtTrans

---

*Report compiled March 2026. Sources include Kaggle competition page, published papers, competition writeups, web search results, and public code repositories.*
