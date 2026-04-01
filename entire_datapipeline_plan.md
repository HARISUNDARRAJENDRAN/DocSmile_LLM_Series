## Data pipeline description

This markdown file describes the strategies and techniques we will use to build the data pipeline for the LLM and VLM.

The project draws on a diverse set of in-depth medical and dental textbooks used as student study materials. The corpus is stored in Google Drive; access the folder here:

https://drive.google.com/drive/folders/1EeJiWpi1jZauvpNcLFBLAoUrtE7bnJB4?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto

We use **Docling** as the primary tool to extract text from PDFs and convert it to Markdown (`.md`). Those converted files feed **continued pre-training (CPT)** of the text LLM.

Docling OCR and CPT are GPU-heavy. To limit spend, we rely on cloud GPUs from **Lightning AI**, **Google Cloud** (e.g. **$300** free trial / credits for new accounts — confirm current terms), **DigitalOcean** (credits may be available via GitHub Education — confirm the current offer; international card verification may be required), and **Azure** (e.g. student or GitHub Education credits — confirm current amount).

### 1. Accessing GPUs with Lightning AI

Lightning AI offers recurring credits (on the order of **$15/month** for many accounts — confirm on the pricing page), which can be used for GPUs such as H100 or A100 depending on availability.

**Steps to create a Lightning AI account and get credits:**

1. Open **https://lightning.ai/**.
2. On the landing page, choose **Create account** (or equivalent).
3. Sign in with **Google** (or another supported provider).
4. Complete onboarding; credits appear on the account according to the current promotion.

Use Lightning AI mainly for **small experiments**: OCR trials, notebook tests, and iteration before running large jobs on GCP.

### 2. Using GCP

GCP is our **primary** source of GPU capacity for production-scale Docling runs, thanks to the **$300** credit (or trial) many new accounts receive — always confirm [Google Cloud’s current signup terms](https://cloud.google.com/).

1. Open **https://console.cloud.google.com/** and create or select a project.
2. Complete **billing** setup. Google may place a **small temporary verification hold** on a payment method (amount and currency vary by **country/region**; it is not always a fixed “₹1000” — check the confirmation screen during signup).
3. After verification, apply the **$300** credit (or equivalent promotion) to eligible services per Google’s terms.
4. Enable the **Compute Engine API** and configure **VPC / firewall** rules as needed (e.g. SSH from your IP if you manage VMs manually).
5. **Recommended VM for Docling at book scale:** **G2** instances with **NVIDIA L4** (e.g. `g2-standard-8` with 1× L4). L4 matches Docling’s published benchmarks and balances cost and throughput. Scale to **multiple L4s** on larger `g2-standard-*` shapes or use **multiple VMs** when parallelizing hundreds of PDFs.
6. **Storage:** Use **Cloud Storage (GCS)** for **input PDFs**, **output Markdown**, and **logs/manifests** — avoids filling VM disks and simplifies **Spot** / **preemptible** retries.
7. **Cost controls:** Prefer **Spot VMs** for batch OCR if interruptions are acceptable; set **budget alerts** in Cloud Billing; run a **500–1000 page pilot** to estimate GPU-hours before processing **~340k pages**.

**GCP role in DocSmile:** Primary environment for **large-scale Docling** conversion and later **CPT** (or orchestration to other compute), using credits so preprocessing stays within budget.

### 3. DigitalOcean (optional)

GitHub Education and similar programs sometimes include **DigitalOcean** credits (e.g. on the order of **$200** — **confirm current terms**). Useful for smaller experiments, CPU-heavy preprocessing, or auxiliary services if GCP is saturated. GPU SKUs and pricing differ from GCP; treat DigitalOcean as **supplemental** unless you validate GPUs and cost for your workload.

### 4. Azure (optional)

**Azure for Students** / **GitHub Education** may include **Azure credits** (e.g. on the order of **$150** — **confirm current terms**). Use for pilots, secondary training runs, or storage/compute mixes. Pick **region** and **GPU SKU** to match PyTorch/CUDA and your framework.
