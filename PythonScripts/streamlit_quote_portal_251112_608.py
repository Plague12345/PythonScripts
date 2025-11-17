from __future__ import annotations
# streamlit_quote_portal_v6.0_with_nerdio.py
# Freedom IT - Pro Quote Portal
# Version 6.0 + NERDIO - FILE UPLOAD, PREVIEW, PAGE ORDERING & NERDIO INTEGRATION:
# - NEW v6.0: Upload additional PDFs and images to merge into proposals
# - NEW v6.0: On-screen preview of all pages with reordering
# - NEW v6.0: Drag-and-drop style page ordering interface
# - NEW v6.0: Automatic PDF merging with custom page order
# - NEW v6.0: Nerdio Manager cost estimate JSON import
# - NEW v6.0: Azure Virtual Desktop cost breakdown in PDFs
# - ALL v5.0 features retained below:
# - Fixed: Company ONBOARDING mention (not offboarding)
# - Improved: Header repetition with better ReportLab parameters
# - Service Accounts and Site Infrastructure support
# - NEW v5.0: Comprehensive HTML parser with support for:
#   * All heading levels (h1-h6)
#   * Ordered lists (<ol>), unordered lists (<ul>)
#   * Definition lists (<dl>, <dt>, <dd>)
#   * Tables with thead/tbody/tfoot and colspan/rowspan
#   * Blockquotes, preformatted text (<pre>, <code>)
#   * Horizontal rules (<hr>)
#   * Styled divs and callout boxes
#   * Inline elements (<span>, <small>, <sup>, <sub>)
#   * Address blocks (<address>)



# --- Hot-fix: ensure template loader exists globally ---
try:
    load_template_markdown  # noqa: F821
except NameError:
    from pathlib import Path
    def load_template_markdown(slug: str, templates_dir: str | Path | None) -> tuple[str, str]:
        """Load a markdown template by slug from the templates directory.
        First line beginning with '# ' is treated as the page title. Returns (title, markdown)."""
        base = Path(templates_dir) if templates_dir else Path("./templates")
        p = base / f"{slug}.md"
        if not p.exists():
            title = slug.replace("_", " ").title()
            md = f"# {title}\n\n_Missing template: {p}_"
            return title, md
        raw = p.read_text(encoding="utf-8").strip()
        lines = raw.splitlines()
        title = lines[0][2:].strip() if lines and lines[0].startswith("# ") else slug.replace("_"," ").title()
        return title, raw
# --- end hot-fix ---

import io
import os
import re
import base64
import json
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

import pandas as pd
import streamlit as st

# === V6.0 NEW: PDF manipulation imports ===
try:
    from PIL import Image
    from PyPDF2 import PdfWriter, PdfReader
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    st.warning("⚠️ For file upload support, install: `pip install PyPDF2 Pillow`")

# === V6.0 NEW: Initialize session state for uploads and ordering ===
if 'uploaded_additional_files' not in st.session_state:
    st.session_state.uploaded_additional_files = []
if 'page_order' not in st.session_state:
    st.session_state.page_order = []
if 'preview_mode' not in st.session_state:
    st.session_state.preview_mode = False
if 'all_pages' not in st.session_state:
    st.session_state.all_pages = []
if 'nerdio_estimate' not in st.session_state:
    st.session_state.nerdio_estimate = None


# ===== Helper Functions =====
def sanitize_emojis_for_pdf(text: str) -> str:
    """
    Replace common emojis with ReportLab-safe symbols.
    ReportLab has limited Unicode support, so we convert emojis to text equivalents.
    """
    replacements = {
        '✅': '[✓]',      # Check mark / Yes
        '❌': '[✗]',      # X mark / No  
        '🔒': '[Secure]',
        '💼': '[Business]',
        '📧': '[Email]',
        '📱': '[Mobile]',
        '🔐': '[Lock]',
        '☁️': '[Cloud]',
        '🌐': '[Web]',
        '📊': '[Chart]',
        '📧': '[Tools]',
        '⚠️': '[!]',
        '✓': '[✓]',      # Simple checkmark
        '✗': '[✗]',      # Simple X
    }
    
    result = text
    for emoji, replacement in replacements.items():
        result = result.replace(emoji, replacement)
    
    return result

# === V6.0 NEW: PDF Processing Helper Function ===
def merge_pdfs_with_order(main_pdf_bytes: bytes, page_order: List[dict]) -> bytes:
    """Merge PDFs according to specified page order."""
    if not PDF_SUPPORT:
        return main_pdf_bytes
    
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.utils import ImageReader
        
        writer = PdfWriter()
        main_reader = PdfReader(io.BytesIO(main_pdf_bytes))
        
        # Process each page in order
        for page_info in page_order:
            if page_info["source"] == "main":
                writer.add_page(main_reader.pages[page_info["page"]])
            elif page_info["source"] == "upload":
                file_data = page_info["file_data"]
                
                if file_data["type"] == "pdf":
                    reader = PdfReader(io.BytesIO(file_data["bytes"]))
                    writer.add_page(reader.pages[page_info["page"]])
                else:  # Image
                    # Convert image to PDF page
                    pdf_buffer = io.BytesIO()
                    c = canvas.Canvas(pdf_buffer, pagesize=letter)
                    
                    img = Image.open(io.BytesIO(file_data["bytes"]))
                    page_width, page_height = letter
                    img_width, img_height = img.size
                    
                    ratio = min((page_width - 100) / img_width, (page_height - 100) / img_height)
                    new_width = img_width * ratio
                    new_height = img_height * ratio
                    x = (page_width - new_width) / 2
                    y = (page_height - new_height) / 2
                    
                    temp_img = io.BytesIO()
                    img.save(temp_img, format="PNG")
                    temp_img.seek(0)
                    
                    c.drawImage(ImageReader(temp_img), x, y, new_width, new_height)
                    c.showPage()
                    c.save()
                    
                    pdf_buffer.seek(0)
                    reader = PdfReader(pdf_buffer)
                    writer.add_page(reader.pages[0])
        
        output = io.BytesIO()
        writer.write(output)
        return output.getvalue()
    except Exception as e:
        st.error(f"Error merging PDFs: {e}")
        return main_pdf_bytes

# === V6.0 NEW: Nerdio Manager Integration ===
class NerdioEstimate:
    """Parser for Nerdio Manager cost estimate JSON exports."""
    
    def __init__(self, json_data: Dict[str, Any]):
        self.raw_data = json_data
        self._parse_data()
    
    def _parse_data(self):
        """Parse JSON into accessible attributes."""
        # Top-level costs
        self.cost_per_user = self.raw_data.get('costEstimate', '$0/user')
        self.cost_per_month = self.raw_data.get('costPerMonth', 0)
        self.cost_per_user_per_month = self.raw_data.get('costPerUserPerMonth', 0)
        
        # Component costs
        self.azure_cost = self.raw_data.get('azureCostPerMonth', 0)
        self.nerdio_cost = self.raw_data.get('nerdioCostPerMonth', 0)
        self.m365_cost = self.raw_data.get('microsoft365CostPerMonth', 0) or 0
        
        # Margins
        self.margin_total = self.raw_data.get('marginPerMonth', 0)
        
        # Parse configuration
        self.config = {}
        if 'configuration' in self.raw_data:
            try:
                self.config = json.loads(self.raw_data['configuration'])
            except:
                pass
        
        # Parse autoscale savings
        self.autoscale = {}
        if 'autoscaleSavings' in self.raw_data:
            try:
                self.autoscale = json.loads(self.raw_data['autoscaleSavings'])
            except:
                pass
        
        # Extract user settings
        self.users_count = 0
        self.pooled_count = 0
        if 'usersSettings' in self.config:
            us = self.config['usersSettings']
            self.users_count = us.get('usersCount', 0)
            self.pooled_count = us.get('pooledCount', 0)
        
        # Extract desktop pools
        self.desktop_pools = []
        if 'desktopsSettings' in self.config:
            ds = self.config['desktopsSettings']
            for pool in ds.get('pooledDesktops', []):
                self.desktop_pools.append({
                    'name': pool.get('name', 'Pool'),
                    'users': pool.get('numberOfUsers', 0),
                    'vm_size': pool.get('desktopsSize', {}).get('name', 'Unknown'),
                    'cores': pool.get('desktopsSize', {}).get('cores', 0),
                    'memory': pool.get('desktopsSize', {}).get('memory', 0),
                    'min_desktops': pool.get('minCountOfDesktops', 0),
                    'max_desktops': pool.get('maxCountOfDesktops', 0),
                    'min_hours': pool.get('minHours', 0),
                    'max_hours': pool.get('maxHours', 0)
                })
        
        # Extract licenses
        self.licenses = []
        if 'licensingOptions' in self.config:
            lo = self.config['licensingOptions']
            for license_type, license_data in lo.items():
                if license_data.get('checked', False):
                    for lic in license_data.get('licenses', []):
                        self.licenses.append({
                            'type': license_type,
                            'plan': lic.get('licensePlan', 'Unknown'),
                            'count': lic.get('licensesCount', 0)
                        })
        
        # Extract region
        self.region = 'Unknown'
        self.currency = 'AUD'
        if 'regionSettings' in self.config:
            rs = self.config['regionSettings']
            self.region = rs.get('region', 'Unknown')
            currency_map = {0: 'USD', 1: 'EUR', 2: 'AUD', 3: 'GBP'}
            self.currency = currency_map.get(rs.get('currency', 2), 'AUD')
    
    def to_markdown(self) -> str:
        """Generate markdown summary."""
        md = []
        md.append("# Azure Virtual Desktop Cost Estimate\n")
        md.append("## Summary\n")
        md.append(f"- **Cost per User:** {self.cost_per_user} per month")
        md.append(f"- **Total Monthly Cost:** ${self.cost_per_month:,.2f}")
        md.append(f"- **Number of Users:** {self.users_count}")
        md.append(f"- **Region:** {self.region}\n")
        
        md.append("## Cost Breakdown\n")
        md.append(f"- **Azure Infrastructure:** ${self.azure_cost:,.2f}")
        md.append(f"- **Nerdio Manager:** ${self.nerdio_cost:,.2f}")
        md.append(f"- **Microsoft 365:** ${self.m365_cost:,.2f}")
        md.append(f"- **Margin:** ${self.margin_total:,.2f}\n")
        
        if self.desktop_pools:
            md.append("## Desktop Pools\n")
            for pool in self.desktop_pools:
                md.append(f"### {pool['name']}")
                md.append(f"- **Users:** {pool['users']}")
                md.append(f"- **VM Size:** {pool['vm_size']}")
                md.append(f"- **Specs:** {pool['cores']} cores, {pool['memory']} GB RAM")
                md.append(f"- **Autoscale:** {pool['min_desktops']}-{pool['max_desktops']} hosts")
                md.append(f"- **Hours:** {pool['min_hours']}h min, {pool['max_hours']}h peak\n")
        
        if self.licenses:
            md.append("## Licenses\n")
            for lic in self.licenses:
                md.append(f"- **{lic['type']}:** {lic['plan']} x {lic['count']}")
        
        if self.autoscale.get('total', 0) > 0:
            md.append(f"\n## Autoscale Savings")
            md.append(f"- **Monthly Savings:** ${self.autoscale['total']:,.2f}")
        
        return "\n".join(md)


def parse_nerdio_json(file_content: str) -> Optional[NerdioEstimate]:
    """Parse Nerdio JSON export."""
    try:
        data = json.loads(file_content)
        return NerdioEstimate(data)
    except Exception as e:
        st.error(f"Failed to parse Nerdio JSON: {e}")
        return None


def discover_markdown_templates(templates_dir) -> list[tuple[str, str]]:
    """
    Discover all .md files in templates directory.
    Returns list of tuples: (friendly_name, stem)
    Excludes standard template files to avoid duplication.
    
    Args:
        templates_dir: Path object pointing to templates directory
        
    Returns:
        List of (display_name, file_stem) tuples
        
    Example:
        [('Business Premium', 'Business_Premium'), ('Marketing Static', 'marketing_static')]
    """
    from pathlib import Path
    
    # Don't include these standard templates as they're handled separately
    excluded_files = {
        'cover', 'executive-summary', 'client-overview', 'plan-comparison',
        'monthly-investment-breakdown', 'support-overview', 'scope-inclusions',
        'assumptions-exclusions', 'onboarding-next-steps', 'terms', 'appendix',
        'support_explained', 'marketing_static', 'readme', 'README'
    }
    
    templates = []
    
    try:
        templates_path = Path(templates_dir)
        
        if not templates_path.exists():
            return templates
            
        for p in sorted(templates_path.glob("*.md")):
            # Skip if in excluded list (case insensitive)
            if p.stem.lower() in excluded_files:
                continue
                
            # Convert filename to friendly display name
            # Business_Premium.md -> "Business Premium"
            # marketing-static.md -> "Marketing Static"
            friendly = p.stem.replace("_", " ").replace("-", " ").title()
            templates.append((friendly, p.stem))
            
    except Exception as e:
        # Fail gracefully - don't break the app if templates can't be scanned
        try:
            import streamlit as st
            st.sidebar.warning(f"Could not scan templates directory: {e}")
        except:
            pass
    
    return templates

# ===== Document Builder Controls (Sidebar) =====
try:
    import streamlit as st
    from pathlib import Path as _TPath
    
    st.sidebar.header("📄 Document Builder")

    # Base sections that are always available
    sections_base = [
        "Cover", "Executive Summary", "Client Overview", "Plan Comparison",
        "Monthly Investment Breakdown", "Support Overview", "Scope & Inclusions",
        "Assumptions & Exclusions", "Onboarding & Next Steps", "Terms", "Appendix"
    ]

    # === DYNAMIC MARKDOWN TEMPLATE DISCOVERY ===
    # Get templates directory from session state
    _templates_dir = _TPath(st.session_state.get("templates_dir", r"C:\FIT-Quoter\templates"))
    
    # Discover all .md files in templates directory
    dynamic_templates = discover_markdown_templates(_templates_dir)
    dynamic_section_names = [name for name, _ in dynamic_templates]
    
    # Combine base sections with discovered markdown templates
    sections_all = sections_base + dynamic_section_names
    
    # Display info about discovered templates in sidebar
    if dynamic_templates:
        with st.sidebar.expander(f"📋 {len(dynamic_templates)} Dynamic Template(s) Found", expanded=False):
            st.caption("These templates were auto-discovered:")
            for name, stem in dynamic_templates:
                st.caption(f"  • {name} ({stem}.md)")
            st.caption("Add new .md files to templates folder to see them here!")
    
    sections_selected = st.sidebar.multiselect(
        "Sections to include (use order of selection):",
        sections_all,
        default=["Cover", "Executive Summary", "Client Overview", "Plan Comparison", 
                 "Monthly Investment Breakdown", "Support Overview", "Onboarding & Next Steps"]
    )
    manual_order = st.sidebar.text_input("Custom order (optional, comma-separated)", value="")

    def _ordered_sections():
        if manual_order.strip():
            wanted = [s.strip() for s in manual_order.split(",")]
            return [s for s in wanted if s in sections_selected]
        return sections_selected

    support_choice = st.sidebar.selectbox(
        "Support description",
        ["Auto (from quote tier)", "Standard", "Flexible", "Premium", "Custom"]
    )
    custom_support_text = ""
    if support_choice == "Custom":
        custom_support_text = st.sidebar.text_area("Custom Support text", height=220)

    # === V6.0 NEW: File Upload Section ===
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔎 Additional Files")
    
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDFs or images",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Merge additional files into your PDF"
    )
    
    if uploaded_files:
        processed_files = []
        for f in uploaded_files:
            file_bytes = f.read()
            file_type = "pdf" if f.type == "application/pdf" else "image"
            processed_files.append({
                "name": f.name,
                "type": file_type,
                "bytes": file_bytes,
                "size_kb": len(file_bytes) / 1024
            })
        st.session_state.uploaded_additional_files = processed_files
        st.sidebar.success(f"✅ {len(processed_files)} file(s)")
        
        with st.sidebar.expander("📋 Files", expanded=False):
            for idx, f in enumerate(processed_files):
                st.caption(f"{idx+1}. {f['name']} ({f['size_kb']:.1f} KB)")
                if f["type"] == "pdf" and PDF_SUPPORT:
                    try:
                        reader = PdfReader(io.BytesIO(f["bytes"]))
                        st.caption(f"   └─ {len(reader.pages)} page(s)")
                    except:
                        pass
    else:
        st.session_state.uploaded_additional_files = []
    
    # Preview mode toggle
    st.sidebar.markdown("---")
    st.session_state.preview_mode = st.sidebar.checkbox(
        "📁 Page preview & ordering",
        value=st.session_state.get("preview_mode", False),
        help="Enable page reordering"
    )
    
    # === V6.0 NEW: Nerdio JSON Import ===
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Nerdio Cost Estimate")
    
    nerdio_file = st.sidebar.file_uploader(
        "Upload Nerdio JSON",
        type=["json", "txt"],
        help="Import AVD cost estimate from Nerdio Manager",
        key="nerdio_uploader"
    )
    
    if nerdio_file:
        try:
            content = nerdio_file.read().decode('utf-8')
            estimate = parse_nerdio_json(content)
            if estimate:
                st.session_state.nerdio_estimate = estimate
                st.sidebar.success(f"✅ {estimate.users_count} users, {estimate.cost_per_user}")
                
                with st.sidebar.expander("📋 Estimate Details", expanded=False):
                    st.caption(f"Monthly: ${estimate.cost_per_month:,.2f}")
                    st.caption(f"Azure: ${estimate.azure_cost:,.2f}")
                    st.caption(f"Nerdio: ${estimate.nerdio_cost:,.2f}")
                    if estimate.desktop_pools:
                        st.caption(f"Pools: {len(estimate.desktop_pools)}")
            else:
                st.sidebar.error("Failed to parse Nerdio JSON")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
    else:
        st.session_state.nerdio_estimate = None
    
    st.sidebar.markdown("---")

    st.sidebar.subheader("Pricing Display")
    show_ex = st.sidebar.checkbox("Show Ex GST", value=True)
    show_inc = st.sidebar.checkbox("Show Inc GST", value=True)
    gst_rate_cfg = st.sidebar.number_input("GST rate", value=0.10, min_value=0.0, max_value=0.25, step=0.01)
    currency = st.sidebar.selectbox("Currency", ["AUD", "NZD", "USD"])
    decimals = st.sidebar.selectbox("Decimal places", [0,1,2], index=2)

    onboarding_mode = st.sidebar.selectbox(
        "Onboarding fee mode",
        ["Show Outright only", "Show Monthlyised only", "Show Both", "Hide"]
    )

    st.sidebar.subheader("Section Options")
    combine_support_and_services = st.sidebar.checkbox(
        "Combine Monthly Services + Monthly Support on one page (when both exist)", value=True
    )
    show_quote_breakdowns = st.sidebar.checkbox("Show per-quote line breakdowns", value=True)
    show_phone_site_totals = st.sidebar.checkbox("Include Site & Phone subtotals in breakdown", value=True)

    st.sidebar.subheader("Layout & Branding")
    brand_variant = st.sidebar.selectbox("Branding variant", ["Freedom IT (Primary)", "Freedom IT (Light)", "Freedom IT (Dark)"])
    page_margins_mm = st.sidebar.slider("Page margins (mm)", min_value=10, max_value=30, value=18)
    table_density = st.sidebar.selectbox("Table density", ["Compact", "Normal", "Roomy"], index=1)
    row_striping = st.sidebar.checkbox("Striped table rows", value=True)
    font_scale = st.sidebar.slider("Font scale", 0.8, 1.3, 1.0, 0.05)

    st.sidebar.subheader("Footer & Disclaimers")
    show_page_numbers = st.sidebar.checkbox("Show page numbers", value=True)
    footer_note = st.sidebar.text_input("Footer note", value="Freedom IT — Turning IT Obstacles Into Business Stepping Stones")
    show_disclaimer = st.sidebar.checkbox("Show disclaimer", value=True)

    st.sidebar.subheader("Presets")
    preset = st.sidebar.selectbox("Load preset", ["None", "Short Pitch", "Detailed Proposal", "Executive Pack"])
    if preset != "None":
        if preset == "Short Pitch":
            sections_selected = ["Cover", "Executive Summary", "Plan Comparison", "Monthly Investment Breakdown", "Support Overview"]
            show_quote_breakdowns = False
            combine_support_and_services = True
        elif preset == "Detailed Proposal":
            sections_selected = sections_all
            show_quote_breakdowns = True
            combine_support_and_services = False
        elif preset == "Executive Pack":
            sections_selected = ["Cover", "Executive Summary", "Client Overview", "Plan Comparison", "Support Overview", "Onboarding & Next Steps"]
    
        # --- Debug & Advanced Options (Sidebar) ---
    st.sidebar.subheader("📁 Debug & Advanced Options")
    
    enable_services_debug = st.sidebar.checkbox(
        "Enable Services Included Debugging",
        value=False,
        help="Shows detailed information about how services are detected from quotes"
    )
    
    # Store in session state for later use
    st.session_state["enable_services_debug"] = enable_services_debug
    
    # --- Toggle for Services Included section (Sidebar) ---
    try:
        _svc_toggle = st.sidebar.checkbox(
            "Show 'Services Included' section",
            value=True,
            help="Toggle the bullet list of included services on quote pages."
        )
        # Store in session state since doc_cfg["sections"] is a list, not a dict
        st.session_state["services_included_toggle"] = bool(_svc_toggle)
    except Exception as _svc_t_err:
        pass


    doc_cfg = {
    "sections": _ordered_sections(),
    "support_choice": support_choice,
    "custom_support_text": custom_support_text,
    "dynamic_templates": dict(dynamic_templates),  # NEW: Store template mappings
    "pricing": {
        "show_ex": show_ex,
        "show_inc": show_inc,
        "gst_rate": gst_rate_cfg,
        "currency": currency,
        "decimals": decimals,
        "onboarding_mode": onboarding_mode,
    },
    "options": {
        "combine_support_and_services": combine_support_and_services,
        "show_quote_breakdowns": show_quote_breakdowns,
        "show_phone_site_totals": show_phone_site_totals,
        "services_included": st.session_state.get("services_included_toggle", True),  # NEW: Store here instead
        "services_debug": st.session_state.get("enable_services_debug", False),  # NEW: Store here too
    },
    "layout": {
        "brand_variant": brand_variant,
        "page_margins_mm": page_margins_mm,
        "table_density": table_density,
        "row_striping": row_striping,
        "font_scale": font_scale,
        "show_page_numbers": show_page_numbers,
        "footer_note": footer_note,
        "show_disclaimer": show_disclaimer,
    },
    "preset": preset,
}

    if doc_cfg["pricing"]["onboarding_mode"] == "Show Outright only":
        show_onboarding_outright = True; show_onboarding_monthly = False
    elif doc_cfg["pricing"]["onboarding_mode"] == "Show Monthlyised only":
        show_onboarding_outright = False; show_onboarding_monthly = True
    elif doc_cfg["pricing"]["onboarding_mode"] == "Show Both":
        show_onboarding_outright = True; show_onboarding_monthly = True
    else:
        show_onboarding_outright = False; show_onboarding_monthly = False

except Exception:
    doc_cfg = {
        "sections": ["Cover", "Executive Summary", "Client Overview", "Plan Comparison", "Monthly Investment Breakdown", "Support Overview", "Onboarding & Next Steps"],
        "support_choice": "Auto (from quote tier)",
        "custom_support_text": "",
        "pricing": {"show_ex": True, "show_inc": True, "gst_rate": 0.10, "currency": "AUD", "decimals": 2, "onboarding_mode": "Show Both"},
        "options": {"combine_support_and_services": True, "show_quote_breakdowns": True, "show_phone_site_totals": True},
        "layout": {"brand_variant": "Freedom IT (Primary)", "page_margins_mm": 18, "table_density": "Normal", "row_striping": True, "font_scale": 1.0, "show_page_numbers": True, "footer_note": "Freedom IT — Turning IT Obstacles Into Business Stepping Stones", "show_disclaimer": True},
        "preset": "None",
    }
    show_onboarding_outright = True
    show_onboarding_monthly = True


# ---------------------------
# Edit Templates (saved to /templates) — Sidebar block with auto-refresh
# ---------------------------
from pathlib import Path as _TPath
import time as _tt

_templates_dir = _TPath(st.session_state.get("templates_dir", r"C:\FIT-Quoter\templates"))

st.sidebar.markdown("### 📝 Edit Templates (saved to /templates)")
st.sidebar.caption(f"Templates folder: {_templates_dir}")

def _folder_signature(_dir: _TPath) -> float:
    try:
        stats = [p.stat().st_mtime for p in _dir.glob("*.md")]
        return sum(stats) + len(stats)
    except Exception:
        return _tt.time()

colA, colB = st.sidebar.columns([1,1])
with colA:
    if st.button("Refresh templates", use_container_width=True):
        try:
            st.cache_data.clear()
        except Exception:
            pass
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()
with colB:
    if st.button("Open Folder", use_container_width=True):
        st.info(f"Open this path in Explorer: {_templates_dir}")

@st.cache_data(show_spinner=False)
def _list_templates(_dir_str: str, sig: float):
    base = _TPath(_dir_str)
    items = []
    for p in sorted(base.glob("*.md")):
        items.append((p.stem.replace("_", " "), p.stem))
    return items

_sig = _folder_signature(_templates_dir)
_template_items = _list_templates(str(_templates_dir), _sig)
_friendly_names = [t[0] for t in _template_items] or ["<no .md files>"]

_sel = st.selectbox("Template", _friendly_names, index=0, key="tmpl_select_dropdown")
_sel_stem = next((stem for name, stem in _template_items if name == _sel), None)

_current_text = ""
_tmpl_path = (_templates_dir / f"{_sel_stem}.md") if _sel_stem else None
if _tmpl_path and _tmpl_path.exists():
    try:
        _current_text = _tmpl_path.read_text(encoding="utf-8")
    except Exception as _e:
        st.sidebar.warning(f"Could not read {_tmpl_path}: {_e}")

_edited = st.sidebar.text_area(
    "Content (Markdown; first line starting with '# ' is used as the title)",
    value=_current_text, height=260, key=f"tmpl_text_{_sel_stem}"
)

colS1, colS2 = st.sidebar.columns([1,1])
with colS1:
    if st.button("Save", use_container_width=True, disabled=not _sel_stem):
        try:
            (_templates_dir / f"{_sel_stem}.md").write_text(_edited, encoding="utf-8")
            st.success("Template saved")
            st.cache_data.clear()
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
        except Exception as _e:
            st.error(f"Save failed: {_e}")
with colS2:
    if st.button("Revert to Default", use_container_width=True, disabled=not _sel_stem):
        st.info("No default template store wired yet. We can add one that copies from /default-templates.")

# ===== Pricing flags from doc_cfg =====
try:
    SHOW_EX_GST = bool(doc_cfg["pricing"]["show_ex"])
    SHOW_INC_GST = bool(doc_cfg["pricing"]["show_inc"])
    GST_RATE = float(doc_cfg["pricing"]["gst_rate"])
    CURRENCY = str(doc_cfg["pricing"]["currency"])
    DECIMALS = int(doc_cfg["pricing"]["decimals"])
except Exception:
    SHOW_EX_GST = True; SHOW_INC_GST = True; GST_RATE = 0.10; CURRENCY = "AUD"; DECIMALS = 2

# ===== Money & Column Helpers =====
def money(val, decimals=None, currency=None):
    try:
        d = DECIMALS if decimals is None else int(decimals)
        c = CURRENCY if currency is None else str(currency)
        fmt = "{:,.%df}" % d
        return f"{c} " + fmt.format(float(val))
    except Exception:
        try:
            return f"{CURRENCY} {float(val):,.2f}"
        except Exception:
            return str(val)

def ex_to_inc(ex_val, gst_rate=None):
    r = GST_RATE if gst_rate is None else float(gst_rate)
    try:
        return float(ex_val) * (1.0 + r)
    except Exception:
        return ex_val

def build_price_cells(ex_amount):
    """Return a list of cells based on flags SHOW_EX_GST/SHOW_INC_GST."""
    cells = []
    if SHOW_EX_GST:
        cells.append(money(ex_amount))
    if SHOW_INC_GST:
        cells.append(money(ex_to_inc(ex_amount)))
    return cells

def price_header_cells():
    hdrs = []
    if SHOW_EX_GST:
        hdrs.append("Ex GST")
    if SHOW_INC_GST:
        hdrs.append("Inc GST")
    return hdrs


# --- Optional controls for Support Description ---
try:
    support_choice = st.selectbox(
        "Support description style",
        ["Auto (from quote tier)", "Standard", "Flexible", "Premium", "Custom"]
    )
    if support_choice in ("Standard", "Flexible", "Premium"):
        quote_summary = quote_summary if 'quote_summary' in globals() else {}
        quote_summary["SupportTier"] = support_choice
    elif support_choice == "Custom":
        custom_support_text = st.text_area("Custom Support Description", height=260)
except Exception:
    pass

import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.colors import black
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle, KeepTogether
, LongTable,
    ListFlowable,
    ListItem, HRFlowable)



# ===== Support Descriptions (Pre-canned + Custom) =====
SUPPORT_DESCRIPTIONS = {
    "Standard": """
## Freedom IT Support — Standard (Remote Only)

Freedom IT’s Standard Support plan provides **unlimited remote assistance** for your team, ensuring fast and efficient resolution of issues without onsite visits.
It’s ideal for businesses with stable networks and systems that can be managed effectively from afar.

### What's Included
- Unlimited **remote helpdesk support** during business hours
- **Proactive system monitoring** and patch management
- **Microsoft 365 user administration** (password resets, access, licenses)
- **Antivirus and security updates** managed centrally
- **Priority response** for urgent or critical incidents

### Why It Matters
- Fast support without travel delays
- Predictable monthly cost
- Keeps your systems secure and running efficiently
""",

    "Flexible": """
## Freedom IT Support — Flexible (Remote + Limited Onsite)

Freedom IT’s Flexible Support plan combines **unlimited remote support** with a limited allowance of **onsite assistance** for hardware, networking, or escalated issues.
It’s designed for organisations that need physical presence occasionally, without the cost of a full onsite plan.

### What's Included
- Unlimited remote helpdesk support
- **Included onsite hours** for escalations or maintenance
- Proactive monitoring and patching
- Backup and endpoint health checks
- Microsoft 365 and security management

### Why It Matters
- Blends efficiency of remote with reliability of onsite
- Predictable coverage for mixed environments
- Maintains high uptime and user satisfaction
""",

    "Premium": """
## Freedom IT Support — Premium (Remote + Onsite)

Freedom IT’s Premium Support plan provides **complete IT coverage** — unlimited remote helpdesk plus **unlimited onsite support** for your users, systems, and infrastructure.
This plan ensures your business receives the fastest response times and the highest service availability.

### What's Included
- Unlimited remote and onsite support
- Scheduled site visits for reviews and maintenance
- 24/7 system monitoring and patching
- Backup verification and disaster recovery checks
- Advanced endpoint protection and reporting

### Why It Matters
- Maximum uptime and minimal disruption
- Dedicated local support presence
- Predictable, all-inclusive IT coverage
"""
}


def render_support_section(elements, styles, support_tier=None, custom_text=None, add_pagebreak=True):
    """Render the Support explainer page without pricing."""
    if custom_text and custom_text.strip():
        support_text = custom_text.strip()
    else:
        key = (support_tier or "Standard").strip().title()
        support_text = SUPPORT_DESCRIPTIONS.get(key, SUPPORT_DESCRIPTIONS["Standard"])

    from reportlab.platypus import Paragraph, Spacer, PageBreak
    from reportlab.lib.units import mm

    if add_pagebreak:
        elements.append(PageBreak())
    elements.append(Paragraph("Freedom IT Support Overview", styles.get("Heading1") or styles.get("Title") or styles["Normal"]))
    elements.append(Spacer(1, 4*mm))
    elements.append(Paragraph(support_text.replace("\\n", "<br/>"), styles.get("Body") or styles["Normal"]))
    elements.append(Spacer(1, 6*mm))

# Render Support Overview using doc_cfg
_tier = None
if doc_cfg.get("support_choice") == "Auto (from quote tier)":
    try:
        _tier = (quote_summary.get("Support Tier") 
                 or quote_summary.get("SupportTier") 
                 or selected_quote.get("Support Tier") 
                 or selected_quote.get("SupportTier"))
    except Exception:
        _tier = None
elif doc_cfg.get("support_choice") in ("Standard", "Flexible", "Premium"):
    _tier = doc_cfg.get("support_choice")
else:
    _tier = None  # Custom

_custom_text = doc_cfg.get("custom_support_text") if doc_cfg.get("support_choice") == "Custom" else None
if 'elements' in globals():
    render_support_section(elements, styles, support_tier=_tier, custom_text=_custom_text, add_pagebreak=True)

from reportlab.lib.utils import ImageReader, simpleSplit

def _do_rerun():
    import streamlit as _st
    if hasattr(_st, 'rerun'):
        _st.rerun()
    elif hasattr(_st, 'experimental_rerun'):
        _st.experimental_rerun()
    else:
        pass

# ---------------------------
# Prefill handling
# ---------------------------
if "prefill_applied" not in st.session_state:
    st.session_state["prefill_applied"] = False

if not st.session_state["prefill_applied"]:
    if "prefill_client_name" in st.session_state:
        st.session_state["client_name"] = st.session_state["prefill_client_name"]
        del st.session_state["prefill_client_name"]
        st.session_state["prefill_applied"] = True

    if "prefill_cover_md" in st.session_state:
        st.session_state["cover_md"] = st.session_state["prefill_cover_md"]
        del st.session_state["prefill_cover_md"]
        st.session_state["prefill_applied"] = True
    
    if st.session_state["prefill_applied"]:
        _do_rerun()


# ---------------------------
# Sidebar-driven style helpers
# ---------------------------
def _style_params_from_cfg(doc_cfg: dict):
    layout = (doc_cfg or {}).get("layout", {})
    density = layout.get("table_density", "Normal")
    pad = {"Compact": 3, "Normal": 6, "Roomy": 9}.get(density, 6)
    row_striping = bool(layout.get("row_striping", True))
    font_scale = float(layout.get("font_scale", 1.0))
    return {"pad": pad, "row_striping": row_striping, "font_scale": font_scale}

def _include_section(doc_cfg: dict, name: str) -> bool:
    try:
        return name in (doc_cfg or {}).get("sections", [])
    except Exception:
        return True
# ---------------------------
# Theme / styles - BRANDED COLOR PALETTE
# ---------------------------
# Brand Colors:
# - Dark Blue: #0018A8 (primary brand color)
# - Light Blue: #5B9BD5 (secondary/accent)
# - Red: #FF0000 (optional highlights)

PRIMARY   = colors.HexColor("#0018A8")  # Dark Blue - main prices, headers, borders
SECONDARY = colors.HexColor("#5B9BD5")  # Light Blue - accents, dividers, highlights
ACCENT    = colors.HexColor("#FF0000")  # Red - optional for totals/callouts
MUTED     = colors.HexColor("#6B7280")  # Grey - secondary text
LIGHT_BG  = colors.HexColor("#F2F6FB")  # 95% Light Blue tint - subtle backgrounds
SUCCESS   = colors.HexColor("#0018A8")  # Dark Blue - GST inclusive prices (brand consistency)
BORDER    = colors.HexColor("#0018A8")  # Dark Blue - borders

styles = getSampleStyleSheet()

styles.add(ParagraphStyle(
    name="CoverTitle", parent=styles["Heading1"], fontSize=36, leading=42,
    alignment=TA_LEFT, spaceBefore=0, spaceAfter=12, textColor=PRIMARY,  # Changed to PRIMARY (Dk Blue)
    fontName="Helvetica-Bold"
))
styles.add(ParagraphStyle(
    name="CoverSub", parent=styles["BodyText"], fontSize=16, leading=22,
    spaceBefore=0, spaceAfter=16, textColor=MUTED
))
styles.add(ParagraphStyle(
    name="CoverClient", parent=styles["BodyText"], fontSize=18, leading=24,
    spaceBefore=4, spaceAfter=4, textColor=PRIMARY, fontName="Helvetica-Bold"
))
styles.add(ParagraphStyle(
    name="Muted", parent=styles["BodyText"], fontSize=9, leading=12,
    textColor=MUTED, spaceBefore=2, spaceAfter=2
))
styles.add(ParagraphStyle(
    name="TitleCentered", parent=styles["Title"], alignment=TA_CENTER,
    textColor=ACCENT, fontSize=24, leading=30, fontName="Helvetica-Bold"
))
styles.add(ParagraphStyle(
    name="H1", parent=styles["Heading1"], textColor=ACCENT,
    fontSize=18, spaceAfter=8, spaceBefore=12, fontName="Helvetica-Bold"
))
styles.add(ParagraphStyle(
    name="H2", parent=styles["Heading2"], textColor=PRIMARY,
    fontSize=14, spaceAfter=6, spaceBefore=8, fontName="Helvetica-Bold"
))
styles.add(ParagraphStyle(
    name="Body", parent=styles["BodyText"], fontSize=10, leading=16
))
styles.add(ParagraphStyle(
    name="Small", parent=styles["BodyText"], fontSize=9, leading=13
))
styles.add(ParagraphStyle(
    name="Price", parent=styles["BodyText"], fontSize=12, leading=18,
    spaceBefore=4, spaceAfter=4
))
styles.add(ParagraphStyle(
    name="SectionIntro", parent=styles["BodyText"], fontSize=11, leading=16,
    textColor=MUTED, spaceBefore=2, spaceAfter=6
))
# v5.0: Additional styles for enhanced HTML support
styles.add(ParagraphStyle(
    name="H3", parent=styles["Heading3"], textColor=PRIMARY,
    fontSize=12, spaceAfter=4, spaceBefore=6, fontName="Helvetica-Bold"
))
styles.add(ParagraphStyle(
    name="H4", parent=styles["Heading4"], textColor=PRIMARY,
    fontSize=11, spaceAfter=3, spaceBefore=5, fontName="Helvetica-Bold"
))
styles.add(ParagraphStyle(
    name="H5", parent=styles["Heading5"], textColor=PRIMARY,
    fontSize=10, spaceAfter=3, spaceBefore=4, fontName="Helvetica-Bold"
))
styles.add(ParagraphStyle(
    name="H6", parent=styles["Heading6"], textColor=PRIMARY,
    fontSize=9, spaceAfter=2, spaceBefore=3, fontName="Helvetica-Bold"
))
styles.add(ParagraphStyle(
    name="BodyIndent", parent=styles["BodyText"], fontSize=10, leading=16,
    leftIndent=20, spaceBefore=2, spaceAfter=2
))
# Code style - check if it already exists (some ReportLab versions include it)
if "Code" not in styles:
    styles.add(ParagraphStyle(
        name="Code", parent=styles["BodyText"], fontSize=9, leading=13,
        fontName="Courier", leftIndent=10, rightIndent=10,
        spaceBefore=4, spaceAfter=4, backColor=colors.HexColor('#F5F5F5')
    ))

# ---------------------------
# Constants
# ---------------------------
MAX_PREVIEW_ROWS = 50
MAX_DESC_LENGTH_FOR_TWO_COL = 600
MAX_SERVICES_FOR_TWO_COL = 15
MAX_POINTS_FOR_TWO_COL = 10
MAX_INCLUDED_POINTS = 10
BULLET_PREFIXES = ("- ", "* ", "• ", "+ ", "- ", "- ")
BUL = "• "

# ---------------------------
# Utilities
# ---------------------------
def escape_html(s: str) -> str:
    if not s:
        return ""
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;")
             .replace("'", "&#x27;"))

def safe_image_flowable(source, max_w: float = 80*mm, max_h: float = 80*mm):
    if source is None:
        return Spacer(1, 1)
    
    try:
        # Handle different input types and get the actual source for Image()
        if hasattr(source, 'getSize') and hasattr(source, '_image'):
            # Already an ImageReader object - extract the underlying image source
            # Don't create another ImageReader, just get dimensions and use original source
            iw, ih = source.getSize()
            # Get the original source from the ImageReader
            if hasattr(source, 'fileName'):
                img_source = source.fileName
            elif hasattr(source, 'fp'):
                img_source = source.fp
            else:
                # Fallback: use the ImageReader itself
                img_source = source
        elif isinstance(source, bytes):
            img_source = io.BytesIO(source)
            rdr = ImageReader(img_source)
            iw, ih = rdr.getSize()
        elif isinstance(source, str):
            if not os.path.exists(source):
                st.warning(f"Logo file not found: {source}")
                return Spacer(1, 1)
            img_source = source
            rdr = ImageReader(source)
            iw, ih = rdr.getSize()
        else:
            return Spacer(1, 1)

        aspect = iw / ih
        
        if iw > max_w or ih > max_h:
            scale_w = max_w / iw
            scale_h = max_h / ih
            scale = min(scale_w, scale_h)
            iw, ih = iw * scale, ih * scale
        
        return Image(img_source, width=iw, height=ih)
    except Exception as e:
        st.warning(f"Failed to load logo: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return Spacer(1, 1)

def fmt_inline(s: str) -> str:
    s = escape_html(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"\*(.+?)\*", r"<i>\1</i>", s)
    s = re.sub(r"`(.+?)`", r"<font name='Courier'>\1</font>", s)
    return s

def md_to_flowables(text: str) -> List[Any]:
    if not text:
        return []

    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\\n", "\n")
    lines = text.split("\n")
    out: List[Any] = []
    para_buf: List[str] = []
    list_buf: List[str] = []
    
    def flush_para():
        nonlocal para_buf
        if para_buf:
            merged = " ".join(para_buf)
            if merged.strip():
                out.append(Paragraph(fmt_inline(merged), styles["Body"]))
            para_buf = []
    
    def flush_list():
        nonlocal list_buf
        if list_buf:
            for itm in list_buf:
                out.append(Paragraph(BUL + fmt_inline(itm), styles["Body"]))
            list_buf = []

    for raw in lines:
        s = raw.strip()
        if s == "":
            flush_list(); flush_para(); out.append(Spacer(1, 4))
            continue

        if s.startswith("# "):
            flush_list(); flush_para()
            out.append(Paragraph(fmt_inline(s[2:]), styles["H1"]))
            continue
        if s.startswith("## "):
            flush_list(); flush_para()
            out.append(Paragraph(fmt_inline(s[3:]), styles["H2"]))
            continue
        if s.startswith("### "):
            flush_list(); flush_para()
            out.append(Paragraph(fmt_inline(s[4:]), styles["H2"]))
            continue

        if s.startswith(BULLET_PREFIXES):
            flush_para()
            item = s[2:].strip() if s[1] == " " else s.lstrip("-•*+--").strip()
            if item:
                list_buf.append(item)
            continue

        flush_list()
        para_buf.append(s)

    flush_list(); flush_para()
    return out


# --- HTML-ish to Flowables (safe subset for ReportLab) ---
def htmlish_to_flowables(html_text: str):
    """
    Comprehensive HTML mapper for ReportLab PDF generation.
    
    SUPPORTED TAGS (v5.0):
    - Headings: <h1>, <h2>, <h3>, <h4>, <h5>, <h6>
    - Paragraphs: <p>
    - Lists: <ul>, <ol>, <li>, <dl>, <dt>, <dd>
    - Tables: <table>, <thead>, <tbody>, <tfoot>, <tr>, <th>, <td> (with colspan/rowspan)
    - Containers: <div>, <blockquote>
    - Text formatting: <span>, <small>, <sup>, <sub>
    - Separators: <hr>
    - Code: <pre>, <code>
    - Address: <address>
    
    Inline tags passed through: <b>, <strong>, <i>, <em>, <u>, <font>, <a>, <br/>
    """
    import re as _re
    story = []
    if not html_text:
        return story

    text = html_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = text.replace("<!-- FORMAT:HTML -->", "").strip()

    # Enhanced regex to capture all supported block-level elements
    blocks = _re.findall(
        r'(<h[1-6][^>]*>.*?</h[1-6]>|'
        r'<p[^>]*>.*?</p>|'
        r'<ul[^>]*>.*?</ul>|'
        r'<ol[^>]*>.*?</ol>|'
        r'<dl[^>]*>.*?</dl>|'
        r'<table[^>]*>.*?</table>|'
        r'<div[^>]*>.*?</div>|'
        r'<blockquote[^>]*>.*?</blockquote>|'
        r'<pre[^>]*>.*?</pre>|'
        r'<address[^>]*>.*?</address>|'
        r'<hr[^>]*>|'
        r'<hr[^>]*/?>)',
        text,
        flags=_re.DOTALL | _re.IGNORECASE
    )

    if not blocks:
        story.append(Paragraph(text, styles["Body"]))
        return story

    def _strip(tag, s):
        """Strip opening and closing tags, handling attributes"""
        return _re.sub(fr'^<{tag}[^>]*>|</{tag}>$', '', s, flags=_re.IGNORECASE).strip()

    def _parse_list(list_html, ordered=False):
        """Parse <ul> or <ol> and return ListFlowable"""
        items = _re.findall(r'<li[^>]*>(.*?)</li>', list_html, flags=_re.DOTALL | _re.IGNORECASE)
        if not items:
            return None
        
        list_items = []
        for idx, item in enumerate(items, 1):
            cleaned = _re.sub(r'\s+', ' ', item).strip()
            bullet = str(idx) + '.' if ordered else '•'
            list_items.append(
                ListItem(Paragraph(cleaned, styles["Body"]), leftIndent=6, bulletText=bullet)
            )
        
        return ListFlowable(list_items, bulletType='bullet', leftIndent=12)

    def _parse_definition_list(dl_html):
        """Parse <dl> and return flowables for definition list"""
        flowables = []
        
        # Extract dt/dd pairs
        dt_pattern = r'<dt[^>]*>(.*?)</dt>'
        dd_pattern = r'<dd[^>]*>(.*?)</dd>'
        
        dts = _re.findall(dt_pattern, dl_html, flags=_re.DOTALL | _re.IGNORECASE)
        dds = _re.findall(dd_pattern, dl_html, flags=_re.DOTALL | _re.IGNORECASE)
        
        # Create term/definition pairs
        for i in range(max(len(dts), len(dds))):
            if i < len(dts):
                term = _re.sub(r'\s+', ' ', dts[i]).strip()
                # Make terms bold
                flowables.append(Paragraph(f"<b>{term}</b>", styles["Body"]))
            if i < len(dds):
                definition = _re.sub(r'\s+', ' ', dds[i]).strip()
                # Indent definitions
                flowables.append(Paragraph(definition, styles["BodyIndent"]))
        
        return flowables

    def _parse_table(table_html):
        """Parse HTML table and convert to ReportLab Table with full support"""
        # Check for thead, tbody, tfoot
        has_sections = bool(_re.search(r'<thead[^>]*>|<tbody[^>]*>|<tfoot[^>]*>', table_html, _re.IGNORECASE))
        
        if has_sections:
            # Extract sections
            thead_match = _re.search(r'<thead[^>]*>(.*?)</thead>', table_html, flags=_re.DOTALL | _re.IGNORECASE)
            tbody_match = _re.search(r'<tbody[^>]*>(.*?)</tbody>', table_html, flags=_re.DOTALL | _re.IGNORECASE)
            tfoot_match = _re.search(r'<tfoot[^>]*>(.*?)</tfoot>', table_html, flags=_re.DOTALL | _re.IGNORECASE)
            
            sections = []
            if thead_match:
                sections.append(('thead', thead_match.group(1)))
            if tbody_match:
                sections.append(('tbody', tbody_match.group(1)))
            if tfoot_match:
                sections.append(('tfoot', tfoot_match.group(1)))
            
            all_rows = []
            section_indices = {'thead': [], 'tbody': [], 'tfoot': []}
            
            for section_type, section_html in sections:
                start_idx = len(all_rows)
                rows = _re.findall(r'<tr[^>]*>(.*?)</tr>', section_html, flags=_re.DOTALL | _re.IGNORECASE)
                for row in rows:
                    all_rows.append((section_type, row))
                end_idx = len(all_rows) - 1
                if start_idx <= end_idx:
                    section_indices[section_type] = list(range(start_idx, end_idx + 1))
        else:
            # No sections, just extract all rows
            rows = _re.findall(r'<tr[^>]*>(.*?)</tr>', table_html, flags=_re.DOTALL | _re.IGNORECASE)
            all_rows = [('tbody', row) for row in rows]
            section_indices = {'thead': [], 'tbody': list(range(len(all_rows))), 'tfoot': []}
        
        # Parse rows into table data
        table_data = []
        col_spans = []  # Track colspan/rowspan info
        
        for section_type, row_html in all_rows:
            cells = _re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row_html, flags=_re.DOTALL | _re.IGNORECASE)
            cell_tags = _re.findall(r'<(t[dh])([^>]*)>', row_html, flags=_re.DOTALL | _re.IGNORECASE)
            
            row_data = []
            for i, cell in enumerate(cells):
                cleaned = _re.sub(r'\s+', ' ', cell).strip()
                
                # Check for colspan/rowspan in cell attributes
                attrs = cell_tags[i][1] if i < len(cell_tags) else ''
                colspan_match = _re.search(r'colspan=["\']?(\d+)["\']?', attrs, _re.IGNORECASE)
                rowspan_match = _re.search(r'rowspan=["\']?(\d+)["\']?', attrs, _re.IGNORECASE)
                
                colspan = int(colspan_match.group(1)) if colspan_match else 1
                rowspan = int(rowspan_match.group(1)) if rowspan_match else 1
                
                # Use different style for headers
                if section_type == 'thead' or cell_tags[i][0].lower() == 'th':
                    row_data.append(Paragraph(f"<b>{cleaned}</b>", styles["Body"]))
                else:
                    row_data.append(Paragraph(cleaned, styles["Body"]))
                
                # Store span info for later styling
                if colspan > 1 or rowspan > 1:
                    col_spans.append((len(table_data), i, colspan, rowspan))
            
            if row_data:
                table_data.append(row_data)
        
        if not table_data:
            return None
        
        # Calculate column widths
        num_cols = max(len(row) for row in table_data) if table_data else 1
        col_width = 160*mm / num_cols
        col_widths = [col_width] * num_cols
        
        # Create table
        tbl = Table(table_data, colWidths=col_widths)
        
        # Build table style
        style_commands = [
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]
        
        # Style thead rows
        if section_indices['thead']:
            for row_idx in section_indices['thead']:
                style_commands.extend([
                    ('BACKGROUND', (0, row_idx), (-1, row_idx), colors.HexColor('#5B9BD5')),
                    ('TEXTCOLOR', (0, row_idx), (-1, row_idx), colors.white),
                    ('FONTNAME', (0, row_idx), (-1, row_idx), 'Helvetica-Bold'),
                ])
        
        # Style tfoot rows
        if section_indices['tfoot']:
            for row_idx in section_indices['tfoot']:
                style_commands.extend([
                    ('BACKGROUND', (0, row_idx), (-1, row_idx), colors.HexColor('#E7E6E6')),
                    ('FONTNAME', (0, row_idx), (-1, row_idx), 'Helvetica-Bold'),
                ])
        
        # Apply colspan/rowspan
        for row, col, colspan, rowspan in col_spans:
            if colspan > 1 or rowspan > 1:
                end_col = col + colspan - 1
                end_row = row + rowspan - 1
                style_commands.append(('SPAN', (col, row), (end_col, end_row)))
        
        # Add grid
        style_commands.append(('GRID', (0, 0), (-1, -1), 0.5, colors.grey))
        
        tbl.setStyle(TableStyle(style_commands))
        return tbl

    def _parse_div(div_html):
        """Parse styled div and convert to ReportLab flowables"""
        style_match = _re.search(r'style="([^"]*)"', div_html, _re.IGNORECASE)
        style_str = style_match.group(1) if style_match else ""
        
        content = _re.sub(r'^<div[^>]*>|</div>$', '', div_html, flags=_re.IGNORECASE).strip()
        
        is_callout = 'background' in style_str.lower() or 'border' in style_str.lower()
        
        # Parse inner content recursively
        inner_blocks = _re.findall(
            r'(<h[1-6][^>]*>.*?</h[1-6]>|<p[^>]*>.*?</p>|<ul[^>]*>.*?</ul>|<ol[^>]*>.*?</ol>|<blockquote[^>]*>.*?</blockquote>)', 
            content,
            flags=_re.DOTALL | _re.IGNORECASE
        )
        
        flowables = []
        
        if is_callout:
            flowables.append(Spacer(1, 2*mm))
        
        if not inner_blocks:
            if content.strip():
                flowables.append(Paragraph(content, styles["Body"]))
        else:
            for inner_block in inner_blocks:
                tag_match = _re.match(r'<(\w+)', inner_block)
                t = tag_match.group(1).lower() if tag_match else ""
                
                if t.startswith('h') and len(t) == 2 and t[1].isdigit():
                    level = int(t[1])
                    style_name = f"H{level}" if f"H{level}" in styles else "H3"
                    flowables.append(Paragraph(_strip(t, inner_block), styles[style_name]))
                elif t == "p":
                    flowables.append(Paragraph(_strip("p", inner_block), styles["Body"]))
                elif t == "ul":
                    lst = _parse_list(inner_block, ordered=False)
                    if lst:
                        flowables.append(lst)
                elif t == "ol":
                    lst = _parse_list(inner_block, ordered=True)
                    if lst:
                        flowables.append(lst)
                elif t == "blockquote":
                    bq_content = _strip("blockquote", inner_block)
                    flowables.append(Paragraph(f"<i>{bq_content}</i>", styles["BodyIndent"]))
        
        if is_callout:
            flowables.append(Spacer(1, 2*mm))
        
        return flowables

    def _parse_blockquote(bq_html):
        """Parse blockquote"""
        content = _strip("blockquote", bq_html)
        # Style blockquotes with italic and indentation
        return [
            Spacer(1, 2*mm),
            Paragraph(f"<i>{content}</i>", styles["BodyIndent"]),
            Spacer(1, 2*mm)
        ]

    def _parse_pre(pre_html):
        """Parse preformatted text / code blocks"""
        content = _strip("pre", pre_html)
        # Remove <code> tags if present
        content = _re.sub(r'</?code[^>]*>', '', content, flags=_re.IGNORECASE)
        # Use Code style if available, otherwise Body with monospace
        code_style = styles.get("Code", styles["Body"])
        return [Paragraph(f"<font face='Courier'>{content}</font>", code_style)]

    # Process blocks
    for b in blocks:
        tag_match = _re.match(r'<(\w+)', b)
        t = tag_match.group(1).lower() if tag_match else ""
        
        # Handle headings (h1-h6)
        if t.startswith('h') and len(t) == 2 and t[1].isdigit():
            level = int(t[1])
            style_name = f"H{level}" if f"H{level}" in styles else "H3"
            story.append(Paragraph(_strip(t, b), styles[style_name]))
        
        # Handle paragraph
        elif t == "p":
            story.append(Paragraph(_strip("p", b), styles["Body"]))
        
        # Handle unordered list
        elif t == "ul":
            lst = _parse_list(b, ordered=False)
            if lst:
                story.append(lst)
        
        # Handle ordered list
        elif t == "ol":
            lst = _parse_list(b, ordered=True)
            if lst:
                story.append(lst)
        
        # Handle definition list
        elif t == "dl":
            flowables = _parse_definition_list(b)
            story.extend(flowables)
        
        # Handle table
        elif t == "table":
            tbl = _parse_table(b)
            if tbl:
                story.append(tbl)
        
        # Handle div
        elif t == "div":
            div_flowables = _parse_div(b)
            story.extend(div_flowables)
        
        # Handle blockquote
        elif t == "blockquote":
            bq_flowables = _parse_blockquote(b)
            story.extend(bq_flowables)
        
        # Handle preformatted text
        elif t == "pre":
            pre_flowables = _parse_pre(b)
            story.extend(pre_flowables)
        
        # Handle address (treat as paragraph with smaller text)
        elif t == "address":
            content = _strip("address", b)
            story.append(Paragraph(f"<font size=9>{content}</font>", styles["Body"]))
        
        # Handle horizontal rule
        elif t == "hr":
            story.append(Spacer(1, 2*mm))
            story.append(HRFlowable(width="100%", thickness=1, color=colors.grey, spaceBefore=2, spaceAfter=2))
            story.append(Spacer(1, 2*mm))
    
    return story
# === Multi-logo wall parser & renderer ===
import re as _logo_re

# Match one logical block of:
# <!-- LOGO_HEADING=... -->
# <!-- LOGO_DIR=... -->
# <!-- LOGO_COLS=... -->
# <!-- LOGO_SIZE_MM=... -->
_LOGO_BLOCK_RE = _logo_re.compile(
    r"<!--\s*LOGO_HEADING=(?P<heading>.*?)\s*-->\s*"
    r"<!--\s*LOGO_DIR=(?P<dir>.*?)\s*-->\s*"
    r"<!--\s*LOGO_COLS=(?P<cols>\d+)\s*-->\s*"
    r"<!--\s*LOGO_SIZE_MM=(?P<size>\d+)\s*-->",
    _logo_re.IGNORECASE | _logo_re.DOTALL
)

def extract_logo_blocks(md_text: str):
    """
    Finds ALL logo blocks, replaces each with a placeholder [[LOGO_BLOCK_n]].
    Returns (cleaned_markdown, list_of_blocks)
    """
    blocks = []

    def _sub(m):
        idx = len(blocks)
        blocks.append({
            "heading": m.group("heading").strip(),
            "logo_dir": m.group("dir").strip(),
            "cols": int(m.group("cols")),
            "size_mm": int(m.group("size")),
        })
        # Wrap in <p> tags so htmlish_to_flowables will capture it
        return f"\n\n<p>[[LOGO_BLOCK_{idx}]]</p>\n\n"

    cleaned = _LOGO_BLOCK_RE.sub(_sub, md_text)
    # Also strip any other HTML comments so they don't show
    cleaned = _logo_re.sub(r"<!--.*?-->", "", cleaned, flags=_logo_re.DOTALL)
    return cleaned.strip(), blocks

def _expand_logo_placeholder(flowable, blocks, styles):
    """
    If the flowable is a Paragraph that contains [[LOGO_BLOCK_n]],
    return a list of flowables that render the heading bar + logo grid.
    Otherwise return None.
    """
    txt = ""
    try:
        txt = flowable.getPlainText().strip()
    except (AttributeError, Exception):
        # Not a Paragraph or doesn't have getPlainText
        return None
    
    # Remove any residual HTML tags that might be in the text
    txt = txt.replace("<p>", "").replace("</p>", "").strip()
    txt = _logo_re.sub(r'<[^>]+>', '', txt).strip()  # Remove any other HTML tags
    
    if not txt:
        return None
    
    # Check if this is a logo block placeholder
    if not (txt.startswith("[[LOGO_BLOCK_") and txt.endswith("]]")):
        return None
    
    m = _logo_re.search(r"\[\[LOGO_BLOCK_(\d+)\]\]", txt)
    if not m:
        return None
    
    idx = int(m.group(1))
    if idx < 0 or idx >= len(blocks):
        return []
    
    blk = blocks[idx]

    parts = []
    # Heading bar
    parts.append(section_banner(blk["heading"], color="#5B9BD5"))
    # Logo grid
    parts.extend(logo_wall(
        blk["logo_dir"],
        cols=blk["cols"],
        row_height=18*mm,
        width_mm=blk["size_mm"]
    ))
    return parts

def to_bullets(text: str) -> List[str]:
    if not text:
        return []

    s = (text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\\n", "\n")
    out: List[str] = []
    
    for raw in s.split("\n"):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith(("what's included", "whats included", "summary")):
            continue

        matched = False
        for pref in BULLET_PREFIXES:
            if line.startswith(pref):
                out.append(line[len(pref):].strip())
                matched = True
                break
        if matched:
            continue

        if ";" in line:
            parts = [p.strip(" -\t") for p in line.split(";")]
            out.extend([p for p in parts if p])

    seen = set()
    cleaned = []
    for p in out:
        if p and p not in seen:
            seen.add(p)
            cleaned.append(p)
    return cleaned

# ---------------------------
# Excel I/O
# ---------------------------
def read_excel_safely(file_bytes: bytes) -> Dict[str, pd.DataFrame]:
    try:
        xls = pd.ExcelFile(io.BytesIO(file_bytes), engine="openpyxl")
        out: Dict[str, pd.DataFrame] = {}
        for name in xls.sheet_names:
            lower = name.strip().lower()
            hdr = None if lower == "quote descriptions" else 0
            df = xls.parse(name, header=hdr)
            df = df.where(pd.notnull(df), "")
            out[lower] = df
        return out
    except Exception as e:
        st.error(f"Failed to read Excel file: {str(e)}")
        return {}

def guess_quote_description_from_excel(sheets: Dict[str, pd.DataFrame]) -> str:
    for key in ("quote summary", "quotesummary"):
        df = sheets.get(key)
        if df is not None and not df.empty:
            for r in range(min(len(df), MAX_PREVIEW_ROWS)):
                row = [str(x).strip() for x in df.iloc[r, :].tolist()]
                texty = [t for t in row if t and not t.replace(",", "").replace(".", "").replace("$", "").isdigit()]
                if len(" ".join(texty)) >= 20:
                    return " ".join(texty)
    ov = sheets.get("overview")
    if ov is not None and not ov.empty:
        col0 = ov.iloc[:,0].astype(str).tolist()
        blob = " ".join([t for t in col0 if t.strip()])
        if len(blob) >= 20:
            return blob
    return ""

def _find_labelled_value(df: pd.DataFrame, labels: list[str]) -> float | None:
    if df is None or df.empty:
        return None
    
    labels_lower = [lbl.lower() for lbl in labels]
    
    for r in range(len(df)):
        try:
            row = df.iloc[r, :]
            row_text = " ".join(str(x).lower() for x in row.to_list())
            
            if any(lbl in row_text for lbl in labels_lower):
                for c in range(1, len(row)):
                    val = row.iat[c]
                    if isinstance(val, (int, float)):
                        if val > 0:
                            return float(val)
                    else:
                        s = str(val).strip().replace("$", "").replace(",", "")
                        if s and s.lower() != "nan":
                            try:
                                num = float(s)
                                if num > 0:
                                    return num
                            except ValueError:
                                continue
        except Exception:
            continue
    return None

def resolve_monthly_support_ex_gst(sheets: Dict[str, pd.DataFrame]) -> float:
    ov = sheets.get("overview")
    
    if ov is not None and not ov.empty:
        for r in range(min(len(ov), 25)):
            try:
                label = str(ov.iat[r, 0]).strip().lower()
                if "support" in label and "monthly" in label:
                    if "sell" in label:
                        v = ov.iat[r, 1]
                        if isinstance(v, (int, float)) and v > 0:
                            return round(float(v), 2)
                        s = str(v).strip().replace("$","").replace(",","")
                        if s and s.lower() != "nan":
                            try:
                                return round(float(s), 2)
                            except ValueError:
                                pass
                    elif "cost" in label:
                        continue
                    elif label == "support monthly" or label == "support monthly (sell)":
                        v = ov.iat[r, 1]
                        if isinstance(v, (int, float)) and v > 0:
                            return round(float(v), 2)
                        s = str(v).strip().replace("$","").replace(",","")
                        if s and s.lower() != "nan":
                            try:
                                return round(float(s), 2)
                            except ValueError:
                                pass
            except Exception:
                continue

    if ov is not None:
        support_labels_sell = [
            "support monthly (sell)",
            "support monthly sell",
            "monthly support sell",
            "support monthly price",
        ]
        v = _find_labelled_value(ov, support_labels_sell)
        if v is not None and v > 0:
            return round(v, 2)

    # REMOVED: Quote Summary row 40 check - was picking up "Monthly Total" instead of "Support Monthly"
    # This was causing ad-hoc quotes to show wrong values in the support field

    label_sets = [
        ["support monthly (sell)", "support monthly sell"],
        ["monthly support (sell)", "monthly support sell"],
        # REMOVED: ["monthly total (ex gst)", "total monthly sell (ex gst)"] - This picks up TOTAL not SUPPORT
    ]
    for nm in ("overview", "quote summary", "quotesummary"):
        df = sheets.get(nm)
        if df is not None:
            for labels in label_sets:
                v = _find_labelled_value(df, labels)
                if v is not None and v > 0:
                    return round(v, 2)

    for _, df in sheets.items():
        ann = _find_labelled_value(df, ["annual support sell", "support annual sell"])
        if ann and ann > 0:
            return round(float(ann)/12.0, 2)
    
    return 0.0

def _find_number_near_labels(df: pd.DataFrame, include_labels: list[str], exclude_labels: list[str] | None = None) -> Optional[float]:
    if df is None or df.empty:
        return None
    
    exclude_labels = [e.lower() for e in (exclude_labels or [])]
    inc = [i.lower() for i in include_labels]
    
    for r in range(len(df)):
        try:
            row = df.iloc[r, :]
            texts = [str(x).strip().lower() for x in row.to_list()]
            joined = " | ".join(texts)
            if any(lbl in joined for lbl in inc):
                if exclude_labels and any(ex in joined for ex in exclude_labels):
                    continue
                nums = pd.to_numeric(row, errors="coerce").dropna()
                if len(nums):
                    return float(nums.max())
        except Exception:
            continue
    return None

def resolve_monthly_services_ex_gst(sheets: Dict[str, pd.DataFrame], support_ex: float | None = None) -> float:
    label_sets = [
        ["monthly services total (ex gst)", "services total (ex gst)", "licensing total (ex gst)"],
        ["monthly services ex gst", "services ex gst"],
    ]
    for nm in ("quote summary", "quotesummary", "overview"):
        df = sheets.get(nm)
        if df is not None:
            for ls in label_sets:
                v = _find_number_near_labels(df, ls, exclude_labels=["support", "labour"])
                if v is not None and v > 0:
                    return round(v, 2)

    generic = None
    for nm in ("quote summary", "quotesummary", "overview"):
        df = sheets.get(nm)
        if df is not None:
            v = _find_number_near_labels(df, ["monthly total (ex gst)", "total monthly ex gst", "monthly ex gst"])
            if v is not None:
                generic = v
                break
    if generic is not None and support_ex is not None and support_ex > 0:
        calc = generic - float(support_ex)
        if calc > 0:
            return round(calc, 2)

    return 0.0

def extract_quote_markdown_from_workbook(sheets: dict) -> str:
    df = sheets.get("quote descriptions")
    if df is None:
        return ""
    try:
        val = df.iat[0, 0]
        if isinstance(val, str) and val.strip():
            return val.replace("\r\n", "\n").replace("\r", "\n").strip()
    except Exception:
        pass
    try:
        col0 = df.columns[0]
        if isinstance(col0, str) and col0.strip() and not col0.lower().startswith("unnamed"):
            return col0.replace("\r\n", "\n").replace("\r", "\n").strip()
    except Exception:
        pass
    return ""
def extract_all_descriptions_from_workbook(sheets: dict) -> dict:
    """Return dict with markdown for 'quote_overview', 'site_services', 'phone_services'.
    Looks in 'Quote Descriptions' sheet with labels in column A and text in column B.
    Label matching is case-insensitive and supports common variants.
    Falls back to extract_quote_markdown_from_workbook for 'quote_overview' if needed.
    """
    result = {"quote_overview": "", "site_services": "", "phone_services": ""}
    df = sheets.get("quote descriptions")
    if df is None or df.empty:
        # Backward compatibility: only a single description available
        result["quote_overview"] = extract_quote_markdown_from_workbook(sheets)
        return result

    # Normalize to two columns
    try:
        # If headers present, keep; otherwise treat first row as data
        if df.shape[1] < 2:
            # Single column fallback: first cell or header becomes overview
            result["quote_overview"] = extract_quote_markdown_from_workbook(sheets)
            return result
        # Create a simple two-column view
        lab_col = df.columns[0]
        val_col = df.columns[1]
        tmp = df[[lab_col, val_col]].copy()
        tmp.columns = ["Label", "Description"]
        tmp = tmp.fillna("")
    except Exception:
        result["quote_overview"] = extract_quote_markdown_from_workbook(sheets)
        return result

    # Label variants
    q_over_labels = {"quote overview","overview","quote description","main description"}
    site_labels   = {"site services","site infrastructure","infrastructure services"}
    phone_labels  = {"phone services","phone","communications","phone & communications"}

    for _, r in tmp.iterrows():
        lab = str(r["Label"]).strip().lower()
        desc = str(r["Description"]).replace("\r\n","\n").replace("\r","\n").strip()
        if not lab or not desc:
            continue
        if lab in q_over_labels and not result["quote_overview"]:
            result["quote_overview"] = desc
        elif lab in site_labels and not result["site_services"]:
            result["site_services"] = desc
        elif lab in phone_labels and not result["phone_services"]:
            result["phone_services"] = desc

    # Fallback for overview
    if not result["quote_overview"]:
        result["quote_overview"] = extract_quote_markdown_from_workbook(sheets)

    return result


def _pick_on_row(df: pd.DataFrame, row_idx: int):
    if df is None or df.empty or row_idx >= len(df):
        return None
    
    try:
        row = df.iloc[row_idx, :]
        for c in range(1, len(row)):
            val = row.iat[c]
            if str(val).strip() and str(val).lower() != "nan":
                return val
    except Exception:
        pass
    return None

def _find_label_value(df: pd.DataFrame, label_variants: list[str]):
    if df is None or df.empty:
        return None
    
    labset = [lv.lower() for lv in label_variants]
    
    try:
        colA = df.iloc[:, 0].astype(str)
        for r, a in enumerate(colA):
            al = a.strip().lower()
            if al in labset:
                return _pick_on_row(df, r)
        for r in range(len(df)):
            row_texts = [str(x).strip().lower() for x in df.iloc[r, :].tolist()]
            row_join = " | ".join(row_texts)
            if any(lbl in row_join for lbl in labset):
                return _pick_on_row(df, r)
    except Exception:
        pass
    
    return None

def _coerce_number(v):
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        n = float(s.replace(",", "").replace("$", ""))
        return int(n) if n.is_integer() else n
    except Exception:
        return None

def extract_client_snapshot(sheets: dict) -> dict:
    """Extract client information including service accounts and phone users support."""
    snap = {
        "client": None, 
        "sites": None, 
        "devices": None, 
        "users": None,
        "phone_users": None,           # Phone users field
        "service_accounts": None,      # Service accounts field
        "proposal_type": "Managed Services + Monthly Services",
        "hourly_rate": None,           # NEW: Hourly rate for ad-hoc support
        "rate_display_type": "monthly" # NEW: Display type (monthly/hourly/managed)
    }
    ov = sheets.get("overview")
    if ov is None or ov.empty:
        return snap

    client           = _find_label_value(ov, ["client name"])
    sites            = _find_label_value(ov, ["number of sites"])
    devices          = _find_label_value(ov, ["number of devices"])
    users            = _find_label_value(ov, ["number of users"])
    phone_users      = _find_label_value(ov, ["number of phone users", "phone users"])
    service_accounts = _find_label_value(ov, ["number of service accounts"])
    tier             = _find_label_value(ov, ["support tier"])
    
    # Try multiple ways to find hourly rate and display type
    # Look in F17-F18 first (where calculated values are cached)
    # Then try helper cells (A36-B37)
    # Then try original labels (D17-E17)
    hourly_rate = None
    display_type = None
    
    # Try to get values directly from specific cells if DataFrame is large enough
    if len(ov) > 16 and len(ov.columns) > 5:
        # Read both F17 and F18 (0-indexed: row 16-17, column 5)
        f17_val = ov.iat[16, 5] if pd.notna(ov.iat[16, 5]) else None
        f18_val = ov.iat[17, 5] if pd.notna(ov.iat[17, 5]) else None
        
        # Treat empty strings as None
        if isinstance(f17_val, str) and not f17_val.strip():
            f17_val = None
        if isinstance(f18_val, str) and not f18_val.strip():
            f18_val = None
        
        # SMART DETECTION: Figure out which cell contains the number vs text
        f17_is_number = isinstance(f17_val, (int, float)) and not isinstance(f17_val, bool)
        f18_is_number = isinstance(f18_val, (int, float)) and not isinstance(f18_val, bool)
        
        if f17_is_number:
            # Layout A: F17 = hourly rate (number), F18 = display type (text)
            hourly_rate = f17_val
            display_type = f18_val
        elif f18_is_number:
            # Layout B: F17 = display type (text), F18 = hourly rate (number)
            display_type = f17_val
            hourly_rate = f18_val
        else:
            # Neither is a number, try to assign text values
            if f17_val is not None and isinstance(f17_val, str):
                display_type = f17_val
            if f18_val is not None and isinstance(f18_val, str):
                hourly_rate = f18_val
    
    # Fallback: search by label (ALWAYS try this if values are still None or invalid)
    if hourly_rate is None or (isinstance(hourly_rate, str) and not hourly_rate.strip()):
        hourly_rate = (_find_label_value(ov, ["hourly rate (pdf)", "hourly rate"]) or
                      _find_label_value(ov, ["hourly rate pdf"]))
    if display_type is None or (isinstance(display_type, str) and not display_type.strip()):
        display_type = _find_label_value(ov, ["rate display type", "display type"])

    snap["client"]           = (str(client).strip() or None) if client is not None else None
    snap["sites"]            = _coerce_number(sites)
    snap["devices"]          = _coerce_number(devices)
    snap["users"]            = _coerce_number(users)
    snap["phone_users"]      = _coerce_number(phone_users)
    snap["service_accounts"] = _coerce_number(service_accounts)
    snap["hourly_rate"]      = _coerce_number(hourly_rate)
    snap["rate_display_type"] = (str(display_type).strip().lower() if display_type is not None else "monthly")

    if tier is not None and str(tier).strip().lower() == "adhoc":
        snap["proposal_type"] = "Ad hoc Support + Monthly Services"
    else:
        snap["proposal_type"] = "Managed Services + Monthly Services"
    return snap

# ---------------------------
# NEW: Site Infrastructure Extraction
# ---------------------------
def extract_site_services(sheets: dict) -> list[dict]:
    """
    Extract site-based services from Core Pricing sheet where Unit Type = 'Site'.
    Returns list of dicts with: {item, unit_price, quantity, total}
    """
    df = sheets.get("core pricing")
    if df is None or df.empty:
        return []
    
    # Normalize column names
    cols = {str(c).strip().lower(): c for c in df.columns}
    
    # Find key columns
    unit_type_col = cols.get("unit type") or cols.get("unittype")
    item_col = cols.get("item")
    customer_name_col = None
    for k, orig in cols.items():
        if k in ("customer product name", "customer name", "display name"):
            customer_name_col = orig
            break
    
    # Price columns - updated to match actual Excel template column names
    unit_price_col = (cols.get("unit price (sell)") or cols.get("unit price sell") or 
                      cols.get("unit price") or cols.get("sell (ex gst)"))
    qty_col = cols.get("qty") or cols.get("quantity") or cols.get("qty (auto)")
    total_col = (cols.get("total (sell)") or cols.get("total sell") or cols.get("total") or
                 cols.get("services monthly (sell) [helper]"))
    
    # Include column
    include_col = None
    for k, orig in cols.items():
        if "include" in k:
            include_col = orig
            break
    
    if not unit_type_col or not item_col:
        return []
    
    site_services = []
    
    for _, r in df.iterrows():
        try:
            # Check if this is a site service
            unit_type = str(r.get(unit_type_col, "")).strip().lower()
            if unit_type != "site":
                continue
            
            # Check if included
            if include_col:
                flag = str(r.get(include_col, "")).strip().lower()
                if flag not in ("true", "yes", "y", "1", "t", "on"):
                    continue
            
            # Get item name (prefer customer name if available)
            if customer_name_col and pd.notna(r.get(customer_name_col)):
                item = str(r.get(customer_name_col, "")).strip()
            else:
                item = str(r.get(item_col, "")).strip()
            
            if not item:
                continue
            
            # Get pricing info
            unit_price = 0.0
            if unit_price_col and pd.notna(r.get(unit_price_col)):
                try:
                    unit_price = float(str(r.get(unit_price_col, 0)).replace("$", "").replace(",", ""))
                except:
                    pass
            
            qty = 1
            if qty_col and pd.notna(r.get(qty_col)):
                try:
                    qty = int(float(str(r.get(qty_col, 1)).replace(",", "")))
                except:
                    pass
            
            total = 0.0
            if total_col and pd.notna(r.get(total_col)):
                try:
                    total = float(str(r.get(total_col, 0)).replace("$", "").replace(",", ""))
                except:
                    total = unit_price * qty
            else:
                total = unit_price * qty
            
            site_services.append({
                "item": item,
                "unit_price": unit_price,
                "quantity": qty,
                "total": total
            })
            
        except Exception:
            continue
    
    return site_services

# ---------------------------
# NEW: Phone Services Extraction
# ---------------------------
def extract_phone_services(sheets: dict) -> list[dict]:
    """
    Extract phone-based services from Core Pricing sheet where Unit Type = 'Phone'.
    Returns list of dicts with: {item, unit_price, quantity, total}
    """
    df = sheets.get("core pricing")
    if df is None or df.empty:
        return []
    
    # Normalize column names
    cols = {str(c).strip().lower(): c for c in df.columns}
    
    # Find key columns
    unit_type_col = cols.get("unit type") or cols.get("unittype")
    item_col = cols.get("item")
    customer_name_col = None
    for k, orig in cols.items():
        if k in ("customer product name", "customer name", "display name"):
            customer_name_col = orig
            break
    
    # Price columns
    unit_price_col = (cols.get("unit price (sell)") or cols.get("unit price sell") or 
                      cols.get("unit price") or cols.get("sell (ex gst)"))
    qty_col = cols.get("qty") or cols.get("quantity") or cols.get("qty (auto)")
    total_col = (cols.get("total (sell)") or cols.get("total sell") or cols.get("total") or
                 cols.get("services monthly (sell) [helper]"))
    
    # Include column
    include_col = None
    for k, orig in cols.items():
        if "include" in k:
            include_col = orig
            break
    
    if not unit_type_col or not item_col:
        return []
    
    phone_services = []
    
    for _, r in df.iterrows():
        try:
            # Check if this is a phone service
            unit_type = str(r.get(unit_type_col, "")).strip().lower()
            if unit_type != "phone":
                continue
            
            # Check if included
            if include_col:
                flag = str(r.get(include_col, "")).strip().lower()
                if flag not in ("true", "yes", "y", "1", "t", "on"):
                    continue
            
            # Get item name (prefer customer name if available)
            if customer_name_col and pd.notna(r.get(customer_name_col)):
                item = str(r.get(customer_name_col, "")).strip()
            else:
                item = str(r.get(item_col, "")).strip()
            
            if not item:
                continue
            
            # Get pricing info
            unit_price = 0.0
            if unit_price_col and pd.notna(r.get(unit_price_col)):
                try:
                    unit_price = float(str(r.get(unit_price_col, 0)).replace("$", "").replace(",", ""))
                except:
                    pass
            
            qty = 1
            if qty_col and pd.notna(r.get(qty_col)):
                try:
                    qty = int(float(str(r.get(qty_col, 1)).replace(",", "")))
                except:
                    pass
            
            total = 0.0
            if total_col and pd.notna(r.get(total_col)):
                try:
                    total = float(str(r.get(total_col, 0)).replace("$", "").replace(",", ""))
                except:
                    total = unit_price * qty
            else:
                total = unit_price * qty
            
            phone_services.append({
                "item": item,
                "unit_price": unit_price,
                "quantity": qty,
                "total": total
            })
            
        except Exception:
            continue
    
    return phone_services

# ---------------------------
# NEW: Microsoft 365 Licensing Extraction
# ---------------------------
def extract_m365_licensing(sheets: dict) -> list[dict]:
    """
    Extract Microsoft 365 licensing from Core Pricing sheet where:
    - Category contains 'Microsoft 365' or 'Microsoft 366'
    - Unit Type = 'User' or 'Service Account'
    - Include? = TRUE
    Returns list of dicts with: {item, unit_price, quantity, total, unit_type}
    """
    df = sheets.get("core pricing")
    if df is None or df.empty:
        return []
    
    # Normalize column names
    cols = {str(c).strip().lower(): c for c in df.columns}
    
    # Find key columns
    category_col = cols.get("category")
    unit_type_col = cols.get("unit type") or cols.get("unittype")
    item_col = cols.get("item")
    
    # Customer name column (optional)
    customer_name_col = None
    for k, orig in cols.items():
        if k in ("customer product name", "customer name", "display name"):
            customer_name_col = orig
            break
    
    # Price columns
    unit_price_col = (cols.get("unit price (sell)") or cols.get("unit price sell") or 
                      cols.get("unit price") or cols.get("sell (ex gst)"))
    qty_col = cols.get("qty") or cols.get("quantity") or cols.get("qty (auto)")
    total_col = (cols.get("total (sell)") or cols.get("total sell") or cols.get("total") or
                 cols.get("services monthly (sell) [helper]"))
    
    # Include column
    include_col = None
    for k, orig in cols.items():
        if "include" in k:
            include_col = orig
            break
    
    if not category_col or not unit_type_col or not item_col:
        return []
    
    m365_licenses = []
    
    for _, r in df.iterrows():
        try:
            # Check if this is a Microsoft 365 item
            category = str(r.get(category_col, "")).strip().lower()
            if "microsoft 365" not in category and "microsoft 366" not in category:
                continue
            
            # Check if it's a User or Service Account license
            unit_type = str(r.get(unit_type_col, "")).strip()
            if unit_type not in ("User", "Service Account"):
                continue
            
            # Check if included
            if include_col:
                flag = str(r.get(include_col, "")).strip().lower()
                if flag not in ("true", "yes", "y", "1", "t", "on"):
                    continue
            
            # Get item name (prefer customer name if available)
            if customer_name_col and pd.notna(r.get(customer_name_col)):
                item = str(r.get(customer_name_col, "")).strip()
            else:
                item = str(r.get(item_col, "")).strip()
            
            if not item:
                continue
            
            # Get pricing info
            unit_price = 0.0
            if unit_price_col and pd.notna(r.get(unit_price_col)):
                try:
                    unit_price = float(str(r.get(unit_price_col, 0)).replace("$", "").replace(",", ""))
                except:
                    pass
            
            qty = 0
            if qty_col and pd.notna(r.get(qty_col)):
                try:
                    qty = int(float(str(r.get(qty_col, 0)).replace(",", "")))
                except:
                    pass
            
            # Skip if quantity is 0
            if qty == 0:
                continue
            
            total = 0.0
            if total_col and pd.notna(r.get(total_col)):
                try:
                    total = float(str(r.get(total_col, 0)).replace("$", "").replace(",", ""))
                except:
                    total = unit_price * qty
            else:
                total = unit_price * qty
            
            m365_licenses.append({
                "item": item,
                "unit_price": unit_price,
                "quantity": qty,
                "total": total,
                "unit_type": unit_type
            })
            
        except Exception:
            continue
    
    return m365_licenses

# ---------------------------
# NEW: Onboarding Fees Extraction
# ---------------------------
def extract_onboarding_fees(sheets: dict) -> tuple[float, float]:
    """
    Extract onboarding fees from Overview sheet.
    Returns (monthlyised_amount, outright_amount) tuple.
    Looks for labels like 'Onboarding Fee (Monthlyised)' or 'Onboarding Fee (Outright)'.
    """
    ov = sheets.get("overview")
    if ov is None or ov.empty:
        st.warning("⚠️ DEBUG: No overview sheet found or it's empty")
        return (0.0, 0.0)
    
    st.info(f"📁 DEBUG: Overview sheet has {len(ov)} rows and {len(ov.columns)} columns")
    
    monthlyised = 0.0
    outright = 0.0
    found_rows = []
    
    # Search for onboarding fee rows
    for r in range(min(len(ov), 50)):  # Check first 50 rows
        try:
            label = str(ov.iat[r, 0]).strip().lower()
            
            # Check if this row contains onboarding fee
            if "onboarding" in label and "fee" in label:
                # Get the value from column B (index 1)
                val = ov.iat[r, 1] if len(ov.columns) > 1 else None
                found_rows.append(f"Row {r}: '{label}' = '{val}'")
                
                # Check if this specific row is for monthlyised or outright
                is_monthlyised = "monthlyised" in label or "monthly" in label
                is_outright = "outright" in label or "upfront" in label or "once" in label
                
                st.info(f"📁 Row {r}: '{label}' → Monthlyised={is_monthlyised}, Outright={is_outright}, Value='{val}'")
                
                # Skip if no value
                if val is None or val == '' or str(val).strip() == '':
                    st.warning(f"⚠️ Row {r}: Skipping empty value")
                    continue
                
                # Try to parse the value
                try:
                    if isinstance(val, (int, float)) and val > 0:
                        parsed_val = float(val)
                    else:
                        s = str(val).strip().replace("$", "").replace(",", "")
                        if s and s.lower() not in ("nan", "e", "error", "#value!", "#n/a", "none"):
                            parsed_val = float(s)
                        else:
                            st.warning(f"⚠️ Row {r}: Invalid value string: '{s}'")
                            continue
                    
                    st.success(f"✅ Row {r}: Parsed ${parsed_val:.2f}")
                    
                    if is_monthlyised and parsed_val > 0:
                        monthlyised = parsed_val
                        st.success(f"✅ Set MONTHLYISED = ${monthlyised:.2f}")
                    elif is_outright and parsed_val > 0:
                        outright = parsed_val
                        st.success(f"✅ Set OUTRIGHT = ${outright:.2f}")
                except (ValueError, TypeError) as e:
                    st.error(f"❌ Row {r}: Parse error - {e}")
                    continue
        except Exception as e:
            st.error(f"❌ Row {r}: Error - {e}")
            continue
    
    # Show summary
    if found_rows:
        st.info(f"📋 Found {len(found_rows)} onboarding row(s):")
        for row in found_rows:
            st.code(row)
    else:
        st.warning("⚠️ No onboarding fee rows found in Overview sheet")
    
    st.info(f"🎯 FINAL: Monthlyised=${monthlyised:.2f}, Outright=${outright:.2f}")
    return (round(monthlyised, 2), round(outright, 2))

# ---------------------------
# Graphics & tables
# ---------------------------
def build_cover_comparison_image() -> bytes:
    try:
        fig, ax = plt.subplots(figsize=(4.8, 2.2), dpi=150)
        labels = ["Security", "Inclusion", "Price"]
        values = [4.5, 3.5, 3.0]
        ax.bar(labels, values)
        ax.set_ylim(0, 5)
        ax.set_ylabel("Level (out of 5)")
        ax.set_title("Summary snapshot")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()
    except Exception as e:
        st.warning(f"Failed to generate cover image: {str(e)}")
        return b""

def make_snapshot_table(snap: dict, doc_cfg: dict | None = None) -> Table:
    """Create snapshot table with service accounts and phone users support."""
    data = [
        ["Client",  snap.get("client") or ""],
        ["Sites",   str(snap.get("sites")) if snap.get("sites") is not None else "-"],
        ["Devices", str(snap.get("devices")) if snap.get("devices") is not None else "-"],
        ["Users",   str(snap.get("users")) if snap.get("users") is not None else "-"],
    ]
    
    # Add phone users row if present
    phone_users = snap.get("phone_users")
    if phone_users is not None and phone_users > 0:
        data.append(["Phone Users", str(phone_users)])
    
    # Add service accounts row if present
    service_accounts = snap.get("service_accounts")
    if service_accounts is not None and service_accounts > 0:
        data.append(["Service Accounts", str(service_accounts)])
    
    data.append(["Proposal Type", snap.get("proposal_type") or ""])
    
    tbl = Table(data, colWidths=[50*mm, 90*mm])
    
    # Get density settings
    p = _style_params_from_cfg(doc_cfg or {})
    
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), PRIMARY),
        ("TEXTCOLOR", (0,0), (0,-1), colors.white),
        ("GRID", (0,0), (-1,-1), 0.5, BORDER),
        ("ROWBACKGROUNDS", (1,0), (1,-1), [colors.white, LIGHT_BG]),
        ("LEFTPADDING", (0,0), (-1,-1), max(6, p["pad"]+2)),
        ("RIGHTPADDING", (0,0), (-1,-1), max(6, p["pad"]+2)),
        ("TOPPADDING", (0,0), (-1,-1), p["pad"]),
        ("BOTTOMPADDING", (0,0), (-1,-1), p["pad"]),
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 10),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN", (1,0), (1,-1), "LEFT"),
    ]))
    return tbl

def extract_included_services(sheets: dict) -> list[tuple[str, str]]:
    """
    Extract included services as list of tuples (category, item_name).
    This is for backwards compatibility with existing code.
    """
    df = sheets.get("core pricing")
    if df is None or df.empty:
        return []

    cols = {str(c).strip().lower(): c for c in df.columns}
    cat_col = cols.get("category", None)
    item_col = cols.get("item", None)
    
    customer_name_col = None
    for k, orig in cols.items():
        if k in ("customer product name", "customer name", "display name"):
            customer_name_col = orig
            break
    
    include_col = None
    for k, orig in cols.items():
        if k in ("include?", "include"):
            include_col = orig
            break
    
    if include_col is None:
        for k, orig in cols.items():
            if "include" in k:
                include_col = orig
                break

    if item_col is None or include_col is None:
        return []

    rows: list[tuple[str, str]] = []
    for _, r in df.iterrows():
        try:
            flag = str(r.get(include_col, "")).strip().lower()
            truthy = flag in ("true","yes","y","1","t","on")
            if truthy:
                cat = str(r.get(cat_col, "")).strip() if cat_col else ""
                
                if customer_name_col and pd.notna(r.get(customer_name_col)):
                    itm = str(r.get(customer_name_col, "")).strip()
                else:
                    itm = str(r.get(item_col, "")).strip()
                
                if itm:
                    rows.append((cat, itm))
        except Exception:
            continue
    
    return rows


def extract_included_services_as_items(sheets: dict) -> list[dict]:
    """
    Extract included services as list of item dictionaries with pricing.
    Returns format compatible with services_from_quote() function.
    
    Returns:
        List of dicts with: {name, price, unit_price, quantity, category}
    """
    df = sheets.get("core pricing")
    if df is None or df.empty:
        return []

    cols = {str(c).strip().lower(): c for c in df.columns}
    
    # Find required columns
    cat_col = cols.get("category", None)
    item_col = cols.get("item", None)
    
    # Find customer/display name column
    customer_name_col = None
    for k, orig in cols.items():
        if k in ("customer product name", "customer name", "display name"):
            customer_name_col = orig
            break
    
    # Find include column
    include_col = None
    for k, orig in cols.items():
        if k in ("include?", "include"):
            include_col = orig
            break
    
    if include_col is None:
        for k, orig in cols.items():
            if "include" in k:
                include_col = orig
                break
    
    # Find price column
    price_col = None
    for k, orig in cols.items():
        if k in ("sell (ex gst)", "sell ex gst", "sell", "unit price"):
            price_col = orig
            break
    
    # Find quantity column (optional)
    qty_col = None
    for k, orig in cols.items():
        if k in ("qty (auto)", "qty", "quantity"):
            qty_col = orig
            break

    if item_col is None or include_col is None:
        return []

    items: list[dict] = []
    for _, r in df.iterrows():
        try:
            flag = str(r.get(include_col, "")).strip().lower()
            truthy = flag in ("true","yes","y","1","t","on")
            if truthy:
                cat = str(r.get(cat_col, "")).strip() if cat_col else ""
                
                # Get item name (prefer customer name if available)
                if customer_name_col and pd.notna(r.get(customer_name_col)):
                    item_name = str(r.get(customer_name_col, "")).strip()
                else:
                    item_name = str(r.get(item_col, "")).strip()
                
                if not item_name:
                    continue
                
                # Get price
                price = 0.0
                if price_col and pd.notna(r.get(price_col)):
                    try:
                        price = float(str(r.get(price_col, 0)).replace("$", "").replace(",", ""))
                    except (ValueError, AttributeError):
                        price = 0.0
                
                # Get quantity
                qty = 1
                if qty_col and pd.notna(r.get(qty_col)):
                    try:
                        qty = int(float(str(r.get(qty_col, 1)).replace(",", "")))
                        if qty < 1:
                            qty = 1
                    except (ValueError, AttributeError):
                        qty = 1
                
                total = price * qty
                
                items.append({
                    "name": item_name,
                    "price": total,
                    "unit_price": price,
                    "quantity": qty,
                    "category": cat
                })
        except Exception:
            continue
    
    return items

def make_services_table(items: list[tuple[str, str]], doc_cfg: dict | None = None) -> Table:
    data = [["Feature Category", "Included Service"]]
    if not items:
        data.append(["-", "No services specified"])
    else:
        for cat, itm in items:
            data.append([cat or "-", itm])

    tbl = Table(data, colWidths=[65*mm, 90*mm])
    sp=_style_params_from_cfg(doc_cfg or {})
    style_list=[
        ("BACKGROUND", (0,0), (-1,0), PRIMARY),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,0), 11),
        ("GRID",       (0,0), (-1,-1), 0.5, BORDER),
        ("LEFTPADDING", (0,0), (-1,-1), max(4, sp["pad"]-2)),
        ("RIGHTPADDING", (0,0), (-1,-1), max(4, sp["pad"]-2)),
        ("TOPPADDING", (0,0), (-1,-1), sp["pad"]),
        ("BOTTOMPADDING", (0,0), (-1,-1), sp["pad"]),
        ("ALIGN", (0,0), (-1,0), "LEFT"),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("FONTSIZE", (0,1), (-1,-1), 9),
    ]
    if sp["row_striping"]:
        style_list.append(("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, LIGHT_BG]))
    tbl.setStyle(TableStyle(style_list))
    return tbl



def make_single_price_box(title: str, ex_price: float, inc_price: float, gst_rate: float, 
                          width: float = 160*mm, doc_cfg: dict | None = None) -> Table:
    """
    Create a single pricing box (for use standalone or in side-by-side layout).
    
    Args:
        title: Box title
        ex_price: Price excluding GST
        inc_price: Price including GST
        gst_rate: GST rate (e.g., 0.10 for 10%)
        width: Width of the box (default 160mm for full width, 77mm for side-by-side)
        doc_cfg: Document configuration for density settings
    """
    # Adjust font sizes based on width
    is_narrow = width < 100*mm
    title_size = 11 if is_narrow else 13
    price_size = 14 if is_narrow else 15
    
    # Get density settings
    p = _style_params_from_cfg(doc_cfg or {})
    
    # Title style
    title_style = ParagraphStyle(
        'PriceBoxTitle',
        parent=styles["Heading2"],
        fontSize=title_size,
        textColor=PRIMARY,
        fontName="Helvetica-Bold",
        spaceAfter=4
    )
    
    # Label style
    label_style = ParagraphStyle(
        'SectionLabel',
        parent=styles["Body"],
        fontSize=9,
        textColor=MUTED,
        spaceBefore=2
    )
    
    # Price style
    price_style = ParagraphStyle(
        'LargePrice',
        parent=styles["Body"],
        fontSize=price_size,
        leading=price_size + 3,
        textColor=PRIMARY,
        fontName="Helvetica-Bold"
    )
    
    # Inc GST style
    inc_style = ParagraphStyle(
        'IncPrice',
        parent=styles["Body"],
        fontSize=9,
        textColor=MUTED
    )
    
    # Build the data
    data = [[Paragraph(f"<b>{escape_html(title.upper())}</b>", title_style)]]
    data.append([Paragraph(f'<b>${ex_price:,.2f}</b> ex GST', price_style)])
    data.append([Paragraph(f'${inc_price:,.2f} inc GST ({int(gst_rate*100)}%)', inc_style)])
    
    # Calculate last row
    last_row = len(data) - 1
    
    # Create table
    tbl = Table(data, colWidths=[width])
    tbl.setStyle(TableStyle([
        # Option B: Modern card with Light Blue accent bar
        ("BACKGROUND", (0,0), (-1,-1), colors.white),
        ("BOX", (0,0), (-1,-1), 2, PRIMARY),
        ("LINEBELOW", (0,0), (-1,0), 2, SECONDARY),
        
        # Left accent bar (4mm Light Blue)
        ("LINEAFTER", (0,0), (0,-1), 4*mm, SECONDARY),
        
        # Dynamic spacing based on density
        ("LEFTPADDING", (0,0), (-1,-1), max(10, p["pad"]+6)),
        ("RIGHTPADDING", (0,0), (-1,-1), max(10, p["pad"]+6)),
        ("TOPPADDING", (0,0), (0,0), max(10, p["pad"]+4)),
        ("BOTTOMPADDING", (0,0), (0,0), p["pad"]),
        ("TOPPADDING", (0,1), (-1,-1), p["pad"]),
        ("BOTTOMPADDING", (0,1), (0,last_row-1), p["pad"]),
        ("BOTTOMPADDING", (0,last_row), (0,last_row), max(10, p["pad"]+4)),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    
    return tbl


def make_price_box(title: str, ex_price: float, inc_price: float, gst_rate: float, 
                   display_type: str = "monthly", hourly_rate: float = None, 
                   support_amount: float = None, doc_cfg: dict | None = None) -> Table:
    """
    Create a pricing box that displays:
    - Monthly Services (always shown)
    - PLUS Hourly Rate (for ad-hoc) OR Monthly Support (for managed)
    
    NEW: Returns side-by-side boxes when hourly rate or support amount present!
    
    Args:
        title: Box title
        ex_price: Monthly services price excluding GST
        inc_price: Monthly services price including GST
        gst_rate: GST rate (e.g., 0.10 for 10%)
        display_type: "monthly", "hourly", or "managed" 
        hourly_rate: Hourly rate to display (for ad-hoc support)
        support_amount: Monthly support amount (for managed agreements)
        doc_cfg: Document configuration for density settings
    """
    # Check if we need side-by-side layout
    has_hourly = display_type.lower() == "hourly" and hourly_rate is not None and hourly_rate > 0
    has_support = display_type.lower() == "managed" and support_amount is not None and support_amount > 0
    
    if has_hourly:
        # SIDE-BY-SIDE: Monthly Services + Hourly Rate
        inc_hourly = hourly_rate * (1 + gst_rate)
        
        left_box = make_single_price_box("Monthly Services", ex_price, inc_price, gst_rate, width=77*mm, doc_cfg=doc_cfg)
        right_box = make_single_price_box("Hourly Rate", hourly_rate, inc_hourly, gst_rate, width=77*mm, doc_cfg=doc_cfg)
        
        # Create side-by-side layout
        side_by_side = Table(
            [[left_box, right_box]], 
            colWidths=[77*mm, 77*mm],
            spaceBefore=0,
            spaceAfter=0
        )
        side_by_side.setStyle(TableStyle([
            ("LEFTPADDING", (0,0), (0,0), 0),  # No padding on left box
            ("RIGHTPADDING", (0,0), (0,0), 3*mm),  # 3mm right padding on left box
            ("LEFTPADDING", (1,0), (1,0), 3*mm),  # 3mm left padding on right box
            ("RIGHTPADDING", (1,0), (1,0), 0),  # No padding on right box
            ("TOPPADDING", (0,0), (-1,-1), 0),
            ("BOTTOMPADDING", (0,0), (-1,-1), 0),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
        ]))
        
        return side_by_side
        
    elif has_support:
        # SIDE-BY-SIDE: Monthly Services + Monthly Support
        inc_support = support_amount * (1 + gst_rate)
        
        left_box = make_single_price_box("Monthly Services", ex_price, inc_price, gst_rate, width=77*mm, doc_cfg=doc_cfg)
        right_box = make_single_price_box("Monthly Support", support_amount, inc_support, gst_rate, width=77*mm, doc_cfg=doc_cfg)
        
        # Create side-by-side layout
        side_by_side = Table(
            [[left_box, right_box]], 
            colWidths=[77*mm, 77*mm],
            spaceBefore=0,
            spaceAfter=0
        )
        side_by_side.setStyle(TableStyle([
            ("LEFTPADDING", (0,0), (0,0), 0),  # No padding on left box
            ("RIGHTPADDING", (0,0), (0,0), 3*mm),  # 3mm right padding on left box
            ("LEFTPADDING", (1,0), (1,0), 3*mm),  # 3mm left padding on right box
            ("RIGHTPADDING", (1,0), (1,0), 0),  # No padding on right box
            ("TOPPADDING", (0,0), (-1,-1), 0),
            ("BOTTOMPADDING", (0,0), (-1,-1), 0),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
        ]))
        
        return side_by_side
        
    else:
        # SINGLE BOX: Just monthly services (full width)
        return make_single_price_box(title, ex_price, inc_price, gst_rate, width=160*mm, doc_cfg=doc_cfg)


def render_per_quote_breakdown(q: Dict[str, Any], site_services: list[dict] | None, phone_services: list[dict] | None, gst_rate: float, doc_cfg: dict | None = None) -> List[Any]:
    """Render a compact per-quote line breakdown table using existing totals, driven by sidebar flags.
    Rows include: Monthly Services, optional Site Services, optional Phone Services, Monthly Support, Subtotal (ex), TOTAL (inc).
    """
    name = q.get("name", "Option")
    ex_services = float(q.get("ex_total", 0.0) or 0.0)
    support_ex = float(q.get("support_ex", 0.0) or 0.0)
    include_site_phone = bool((doc_cfg or {}).get("options", {}).get("show_phone_site_totals", True))
    site_ex = sum_monthly_total(site_services) if (include_site_phone and site_services) else 0.0
    phone_ex = sum_monthly_total(phone_services) if (include_site_phone and phone_services) else 0.0

    headers = [f"{name} — Line breakdown", "Amount"]
    data = [headers]
    data.append(["Monthly Services (ex GST)", money(ex_services)])
    if include_site_phone and site_services:
        data.append(["Site Services (ex GST)", money(site_ex)])
    if include_site_phone and phone_services:
        data.append(["Phone Services (ex GST)", money(phone_ex)])
    data.append(["Monthly Support (ex GST)", money(support_ex)])
    data.append(["", ""])
    subtotal_ex = ex_services + site_ex + phone_ex + support_ex
    data.append(["Subtotal (ex GST)", money(subtotal_ex)])
    data.append(["TOTAL (inc GST)", money(ex_to_inc(subtotal_ex, gst_rate))])

    col_widths = [120*mm, 40*mm]
    tbl = Table(data, colWidths=col_widths, repeatRows=1, splitByRow=True)
    p = _style_params_from_cfg(doc_cfg or {})
    style_list = [
        ("BACKGROUND", (0,0), (-1,0), PRIMARY),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("GRID", (0,0), (-1,-1), 0.5, BORDER),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), max(4, p["pad"]-2)),
        ("RIGHTPADDING", (0,0), (-1,-1), max(4, p["pad"]-2)),
        ("TOPPADDING", (0,0), (-1,-1), p["pad"]),
        ("BOTTOMPADDING", (0,0), (-1,-1), p["pad"]),
        ("ALIGN", (1,1), (1,-1), "RIGHT"),
        ("FONTSIZE", (0,0), (-1,0), 10),
        ("FONTSIZE", (0,1), (-1,-1), 10),
    ]

    for i, row in enumerate(data):
        if row[0].startswith("Subtotal"):
            style_list.extend([("BACKGROUND", (0,i), (-1,i), LIGHT_BG), ("FONTNAME", (0,i), (-1,i), "Helvetica-Bold")])
        if row[0].startswith("TOTAL"):
            style_list.extend([("BACKGROUND", (0,i), (-1,i), SUCCESS), ("TEXTCOLOR", (0,i), (-1,i), colors.white), ("FONTNAME", (0,i), (-1,i), "Helvetica-Bold")])

    if p["row_striping"]:
        style_list.append(("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, LIGHT_BG]))
    tbl.setStyle(TableStyle(style_list))
    return [tbl, Spacer(1, 6*mm)]



def sum_monthly_total(items: list[dict]) -> float:
    total = 0.0
    for r in (items or []):
        try:
            v = float(r.get("total", 0.0) or 0.0)
            total += v
        except Exception:
            pass
    return round(total, 2)

def make_monthly_totals_by_quote(quotes: List[Dict[str, Any]], site_ex: float, phone_ex: float, gst_rate: float, doc_cfg: dict | None = None) -> List[Any]:
    """
    Build a per-quote totals table that clearly separates:
    - Monthly Services (varies per quote)
    - Site Services (constant across quotes in this proposal)
    - Phone Services (constant across quotes in this proposal)
    - Monthly Support (varies per quote - only shown if quote has support)
    Shows subtotal ex GST and total inc GST per quote.
    
    IMPROVEMENTS in v42:
    - CORRECTED: Company ONBOARDING mention (not offboarding) 
    - Added repeatRows=1 and splitByRow=True for header repetition on page splits
    - Added company onboarding note at the bottom of the table
    """
    headers = ["Monthly Investment Breakdown"] + [q.get("name", f"Option {i+1}").replace(" - Monthly Services", "") for i, q in enumerate(quotes)]
    data = [headers]

    # Rows: Monthly Services (per quote)
    row_services = ["Monthly Services (ex GST)"]
    for q in quotes:
        ex = float(q.get("ex_total", 0.0) or 0.0)
        row_services.append(f"${ex:,.2f}")
    data.append(row_services)

    # Site, Phone (same across columns)
    data.append(["Site Services (ex GST)"]   + [f"${site_ex:,.2f}"]   * len(quotes))
    data.append(["Phone Services (ex GST)"]  + [f"${phone_ex:,.2f}"]  * len(quotes))
    
    # FIXED: Monthly Support varies per quote - show per-quote amount or "-"
    row_support = ["Monthly Support (ex GST)"]
    for q in quotes:
        q_support = float(q.get("support_ex", 0.0) or 0.0)
        if q_support > 0:
            row_support.append(f"${q_support:,.2f}")
        else:
            row_support.append("-")
    data.append(row_support)

    # Blank spacer
    data.append([""] * len(headers))

    # Subtotals ex GST and TOTAL inc GST
    row_sub_ex = ["Subtotal (ex GST)"]
    row_total_inc = ["TOTAL (inc GST)"]
    for q in quotes:
        ex_services = float(q.get("ex_total", 0.0) or 0.0)
        q_support = float(q.get("support_ex", 0.0) or 0.0)
        subtotal_ex = ex_services + site_ex + phone_ex + q_support
        total_inc = subtotal_ex * (1 + gst_rate)
        row_sub_ex.append(f"${subtotal_ex:,.2f}")
        row_total_inc.append(f"${total_inc:,.2f}")
    data.append(row_sub_ex)
    data.append(row_total_inc)
    
    # CORRECTED in v42: Add company ONBOARDING note (not offboarding)
    data.append([""] * len(headers))  # Spacer
    data.append(["Company Onboarding*"] + ["-"] * len(quotes))

    # Build table with smart split control
    label_width = 70*mm
    value_width = (170*mm - label_width) / len(quotes)
    col_widths = [label_width] + [value_width] * len(quotes)

    # IMPROVED in v41: Always use repeatRows=1 and splitByRow=True for better header repetition
    tbl = Table(data, colWidths=col_widths, repeatRows=1, splitByRow=True)
    
    p=_style_params_from_cfg(doc_cfg or {})
    style_list = [
        ("BACKGROUND", (0,0), (-1,0), PRIMARY),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("GRID", (0,0), (-1,-1), 0.5, BORDER),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), max(4, p["pad"]-2)),
        ("RIGHTPADDING", (0,0), (-1,-1), max(4, p["pad"]-2)),
        ("TOPPADDING", (0,0), (-1,-1), p["pad"]),
        ("BOTTOMPADDING", (0,0), (-1,-1), p["pad"]),
        ("ALIGN", (1,0), (-1,-1), "CENTER"),
        ("ALIGN", (0,0), (0,-1), "LEFT"),
        ("FONTSIZE", (0,0), (-1,0), 10),
        ("FONTSIZE", (0,1), (-1,-1), 10),
    ]

    # Accents
    for i, row in enumerate(data):
        if row and isinstance(row[0], str) and row[0].startswith("Subtotal"):
            style_list.append(("BACKGROUND", (0,i), (-1,i), LIGHT_BG))
            style_list.append(("FONTNAME", (0,i), (-1,i), "Helvetica-Bold"))
        if row and isinstance(row[0], str) and row[0].startswith("TOTAL"):
            style_list.append(("BACKGROUND", (0,i), (-1,i), SUCCESS))
            style_list.append(("TEXTCOLOR", (0,i), (-1,i), colors.white))
            style_list.append(("FONTNAME", (0,i), (-1,i), "Helvetica-Bold"))
            style_list.append(("FONTSIZE", (0,i), (-1,i), 11))
        # CORRECTED in v42: Style company onboarding row (not offboarding)
        if row and isinstance(row[0], str) and row[0].startswith("Company Onboarding"):
            style_list.append(("BACKGROUND", (0,i), (-1,i), LIGHT_BG))
            style_list.append(("FONTNAME", (0,i), (-1,i), "Helvetica-Bold"))
            style_list.append(("FONTSIZE", (0,i), (-1,i), 9))
            style_list.append(("TEXTCOLOR", (0,i), (-1,i), MUTED))

    if p["row_striping"]:
        style_list.append(("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, LIGHT_BG]))
    tbl.setStyle(TableStyle(style_list))
    
    # CORRECTED in v42: Add footnote about company ONBOARDING (not offboarding)
    # Return table with footnote instead of wrapping in KeepTogether
    return [tbl, Spacer(1, 6*mm)]

def make_comprehensive_comparison_table(quotes: List[Dict[str, Any]], gst_rate: float, monthly_support_ex: float, doc_cfg: dict | None = None) -> List[Any]:
    """
    IMPROVED in v41: Always uses repeatRows=1 and splitByRow=True for better header repetition
    when comparison tables span multiple pages.
    """
    elements = []
    all_services = {}
    for q in quotes:
        services = q.get("services_included", [])
        for cat, item in services:
            if cat not in all_services:
                all_services[cat] = {}
            if item not in all_services[cat]:
                all_services[cat][item] = 0
            all_services[cat][item] += 1
    
    headers = ["Plan Inclusions"] + [q.get("name", f"Option {i+1}").replace(" - Monthly Services", "").replace("Option ", "Plan ") for i, q in enumerate(quotes)]
    data = [headers]
    
    def get_category_score(cat_items, doc_cfg: dict | None = None):
        return sum(cat_items.values()) / len(cat_items) if cat_items else 0
    
    sorted_categories = sorted(all_services.keys(), key=lambda c: get_category_score(all_services[c]), reverse=True)
    
    for category in sorted_categories:
        if category:
            cat_row = [category]
            for _ in quotes:
                cat_row.append("")
            data.append(cat_row)
        
        sorted_items = sorted(all_services[category].items(), key=lambda x: x[1], reverse=True)
        
        for item, _ in sorted_items:
            item_row = [f"  {item}" if category else item]
            for q in quotes:
                services = q.get("services_included", [])
                service_items = [itm for cat, itm in services]
                if item in service_items:
                    item_row.append("•")
                else:
                    item_row.append("")
            data.append(item_row)
    
    data.append([""] * len(headers))
    
    # Respect sidebar flags for price display
    _show_ex = (doc_cfg or {}).get("pricing", {}).get("show_ex", True)
    _show_inc = (doc_cfg or {}).get("pricing", {}).get("show_inc", True)
    if _show_ex:
        price_ex_row = ["Monthly Investment (ex GST)"]
        for q in quotes:
            ex = float(q.get("ex_total", 0.0))
            price_ex_row.append(f"${ex:,.2f}")
        data.append(price_ex_row)
    if _show_inc:
        price_inc_row = ["Monthly Investment (inc GST)"]
        for q in quotes:
            ex = float(q.get("ex_total", 0.0))
            inc = gst_inclusive(ex, gst_rate)
            price_inc_row.append(f"${inc:,.2f}")
        data.append(price_inc_row)
    
    # FIXED: Check if ANY quote has support before adding support rows
    has_any_support = any(float(q.get("support_ex", 0.0) or 0.0) > 0 for q in quotes)
    
    if has_any_support:
        data.append([""] * len(headers))
        _show_ex = (doc_cfg or {}).get("pricing", {}).get("show_ex", True)
        _show_inc = (doc_cfg or {}).get("pricing", {}).get("show_inc", True)
        if _show_ex:
            support_ex_row = ["Monthly Support (ex GST)"]
        if _show_inc:
            support_inc_row = ["Monthly Support (inc GST)"]
        for q in quotes:
            q_support_ex = float(q.get("support_ex", 0.0) or 0.0)
            if _show_ex:
                support_ex_row.append(f"${q_support_ex:,.2f}" if q_support_ex>0 else "-")
            if _show_inc:
                support_inc_row.append(f"${gst_inclusive(q_support_ex, gst_rate):,.2f}" if q_support_ex>0 else "-")
        if _show_ex:
            data.append(support_ex_row)
        if _show_inc:
            data.append(support_inc_row)
        data.append([""] * len(headers))
        total_row = ["TOTAL Monthly Investment (inc GST)"]
        for q in quotes:
            ex = float(q.get("ex_total", 0.0))
            q_support_ex = float(q.get("support_ex", 0.0) or 0.0)
            services_inc = gst_inclusive(ex, gst_rate)
            support_inc = gst_inclusive(q_support_ex, gst_rate)
            total = services_inc + support_inc
            total_row.append(f"${total:,.2f}")
        data.append(total_row)
    
    label_width = 70*mm
    value_width = (170*mm - label_width) / len(quotes)
    col_widths = [label_width] + [value_width] * len(quotes)
    
    # IMPROVED in v41: Always use repeatRows=1 and splitByRow=True for better header repetition
    tbl = Table(data, colWidths=col_widths, repeatRows=1, splitByRow=True)
    
    p=_style_params_from_cfg(doc_cfg or {})
    style_list = [
        ("BACKGROUND", (0,0), (-1,0), PRIMARY),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,0), 10),
        ("ALIGN", (1,0), (-1,0), "CENTER"),
        ("GRID", (0,0), (-1,-1), 0.5, BORDER),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), max(4, p["pad"]-2)),
        ("RIGHTPADDING", (0,0), (-1,-1), max(4, p["pad"]-2)),
        ("TOPPADDING", (0,0), (-1,-1), p["pad"]),
        ("BOTTOMPADDING", (0,0), (-1,-1), p["pad"]),
        ("ALIGN", (1,1), (-1,-1), "CENTER"),
        ("FONTSIZE", (1,1), (-1,-1), 12),
        ("ALIGN", (0,1), (0,-1), "LEFT"),
        ("FONTSIZE", (0,1), (0,-1), 9),
    ]
    
    pricing_start = None
    for i, row in enumerate(data):
        if row[0] == "Monthly Investment (ex GST)":
            pricing_start = i
            break
    
    if pricing_start:
        for i in range(1, pricing_start):
            if data[i][0] and not data[i][0].startswith("  "):
                style_list.append(("BACKGROUND", (0,i), (0,i), LIGHT_BG))
                style_list.append(("FONTNAME", (0,i), (0,i), "Helvetica-Bold"))
                style_list.append(("SPAN", (0,i), (-1,i)))
        
        style_list.extend([
            ("BACKGROUND", (0,pricing_start), (0,-1), LIGHT_BG),
            ("FONTNAME", (0,pricing_start), (0,-1), "Helvetica-Bold"),
            ("FONTNAME", (1,pricing_start), (-1,-1), "Helvetica-Bold"),
            ("FONTSIZE", (0,pricing_start), (-1,-1), 10),
            ("TEXTCOLOR", (1,pricing_start), (-1,-1), PRIMARY),
        ])
        
        for i in range(pricing_start, len(data)):
            if data[i][0].startswith("TOTAL"):
                style_list.extend([
                    ("BACKGROUND", (0,i), (-1,i), SUCCESS),
                    ("TEXTCOLOR", (0,i), (-1,i), colors.white),
                    ("FONTSIZE", (0,i), (-1,i), 11),
                ])
                break
    
    if p["row_striping"]:
        style_list.append(("ROWBACKGROUNDS", (0,1), (-1,-3), [colors.white, LIGHT_BG]))
    tbl.setStyle(TableStyle(style_list))
    
    # IMPROVED in v41: Return table without KeepTogether to allow natural splitting with repeated headers
    return [tbl]

# ---------------------------
# NEW: Site Infrastructure Page
# ---------------------------
def site_infrastructure_page(site_services: list[dict], num_sites: int, gst_rate: float, site_desc_md: str = "", doc_cfg: dict | None = None) -> List[Any]:
    """
    Create the Site Infrastructure & Services page (Phase 1).
    Shows all site-based services with per-site pricing.
    """
    elements = []
    
    if not site_services or num_sites is None or num_sites <= 0:
        return elements
    
    # Page title
    elements.append(Paragraph("Site Infrastructure & Services", styles["H1"]))
    title_line = Table([[""]], colWidths=[160*mm])
    title_line.setStyle(TableStyle([
        ("LINEBELOW", (0,0), (-1,0), 2, PRIMARY),
        ("TOPPADDING", (0,0), (-1,-1), 2),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4)
    ]))
    elements.append(title_line)
    elements.append(Spacer(1, 4*mm))
    
    # Intro / description
    if site_desc_md.strip():
        for p in md_to_flowables(site_desc_md):
            elements.append(p)
    else:
        intro_text = f"Infrastructure and services deployed across your {num_sites} {'location' if num_sites == 1 else 'locations'}."
        elements.append(Paragraph(intro_text, styles["SectionIntro"]))
    elements.append(Spacer(1, 6*mm))
    
    # Build table data
    table_data = [["Service", "Per Site", "Sites", "Monthly Total"]]
    
    total_monthly = 0.0
    for svc in site_services:
        item = svc["item"]
        unit_price = svc["unit_price"]
        total = svc["total"]
        
        table_data.append([
            item,
            f"${unit_price:,.2f}",
            str(num_sites),
            f"${total:,.2f}"
        ])
        total_monthly += total
    
    # Add total row
    table_data.append(["", "", "", ""])  # Spacer
    table_data.append([
        "TOTAL Site Services (ex GST)",
        "",
        "",
        f"${total_monthly:,.2f}"
    ])
    
    total_inc_gst = gst_inclusive(total_monthly, gst_rate)
    table_data.append([
        "TOTAL Site Services (inc GST)",
        "",
        "",
        f"${total_inc_gst:,.2f}"
    ])
    
    # Create table
    col_widths = [80*mm, 30*mm, 20*mm, 30*mm]
    tbl = Table(table_data, colWidths=col_widths)
    
    p=_style_params_from_cfg(doc_cfg or {})
    style_list = [
        ("BACKGROUND", (0,0), (-1,0), PRIMARY),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,0), 11),
        ("GRID", (0,0), (-1,-2), 0.5, BORDER),
        ("ROWBACKGROUNDS", (0,1), (-1,-3), [colors.white, LIGHT_BG]),
        ("LEFTPADDING", (0,0), (-1,-1), max(4, p["pad"]-2)),
        ("RIGHTPADDING", (0,0), (-1,-1), max(4, p["pad"]-2)),
        ("TOPPADDING", (0,0), (-1,-1), p["pad"]),
        ("BOTTOMPADDING", (0,0), (-1,-1), p["pad"]),
        ("ALIGN", (1,0), (-1,-1), "CENTER"),
        ("ALIGN", (0,0), (0,-1), "LEFT"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("FONTSIZE", (0,1), (-1,-3), 9),
    ]
    
    # Style total rows
    total_start = len(table_data) - 2
    style_list.extend([
        ("BACKGROUND", (0,total_start), (-1,-1), LIGHT_BG),
        ("FONTNAME", (0,total_start), (-1,-1), "Helvetica-Bold"),
        ("FONTSIZE", (0,total_start), (-1,-1), 10),
        ("TEXTCOLOR", (3,total_start), (3,-1), PRIMARY),
        ("SPAN", (0,total_start), (2,total_start)),
        ("SPAN", (0,total_start+1), (2,total_start+1)),
        ("GRID", (0,total_start), (-1,-1), 0.5, BORDER),
    ])
    
    # Highlight final total
    style_list.extend([
        ("BACKGROUND", (0,-1), (-1,-1), SUCCESS),
        ("TEXTCOLOR", (0,-1), (-1,-1), colors.white),
        ("FONTSIZE", (0,-1), (-1,-1), 11),
    ])
    
    if p["row_striping"]:
        style_list.append(("ROWBACKGROUNDS", (0,1), (-1,-3), [colors.white, LIGHT_BG]))
    tbl.setStyle(TableStyle(style_list))
    # Wrap table in KeepTogether to prevent splitting across pages
    elements.append(KeepTogether([tbl, Spacer(1, 8*mm)]))
    
    # Add explanation section
    elements.append(Paragraph("About Site Infrastructure Services", styles["H2"]))
    elements.append(Spacer(1, 2*mm))
    
    explanation = [
        "Site-based services scale with your number of locations, not your user count",
        "Each site receives the full service independently",
        "Pricing shown is per site per month",
        "Common examples: Internet connectivity, phone systems, firewall management, physical security",
    ]
    
    for point in explanation:
        elements.append(Paragraph(BUL + escape_html(point), styles["Body"]))
    
    elements.append(PageBreak())
    return elements

# ---------------------------
# NEW: Phone Services Page
# ---------------------------
def phone_services_page(phone_services: list[dict], num_phone_users: int, gst_rate: float, phone_desc_md: str = "", doc_cfg: dict | None = None) -> List[Any]:
    """
    Create the Phone Services & Communications page.
    Shows all phone-based services with per-user pricing.
    """
    elements = []
    
    if not phone_services or num_phone_users is None or num_phone_users <= 0:
        return elements
    
    # Page title
    elements.append(Paragraph("Phone Services & Communications", styles["H1"]))
    title_line = Table([[""]], colWidths=[160*mm])
    title_line.setStyle(TableStyle([
        ("LINEBELOW", (0,0), (-1,0), 2, PRIMARY),
        ("TOPPADDING", (0,0), (-1,-1), 2),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4)
    ]))
    elements.append(title_line)
    elements.append(Spacer(1, 4*mm))
    
    # Intro / description
    if phone_desc_md.strip():
        for p in md_to_flowables(phone_desc_md):
            elements.append(p)
    else:
        intro_text = f"Cloud phone services for {num_phone_users} team {'member' if num_phone_users == 1 else 'members'} requiring phone access."
        elements.append(Paragraph(intro_text, styles["SectionIntro"]))
    elements.append(Spacer(1, 6*mm))
    
    # Build table data
    table_data = [["Service", "Per User", "Phone Users", "Monthly Total"]]
    
    total_monthly = 0.0
    for svc in phone_services:
        item = svc["item"]
        unit_price = svc["unit_price"]
        total = svc["total"]
        
        table_data.append([
            item,
            f"${unit_price:,.2f}",
            str(num_phone_users),
            f"${total:,.2f}"
        ])
        total_monthly += total
    
    # Add total row
    table_data.append(["", "", "", ""])  # Spacer
    table_data.append([
        "TOTAL Phone Services (ex GST)",
        "",
        "",
        f"${total_monthly:,.2f}"
    ])
    
    total_inc_gst = gst_inclusive(total_monthly, gst_rate)
    table_data.append([
        "TOTAL Phone Services (inc GST)",
        "",
        "",
        f"${total_inc_gst:,.2f}"
    ])
    
    # Create table
    col_widths = [80*mm, 30*mm, 20*mm, 30*mm]
    tbl = Table(table_data, colWidths=col_widths)
    
    p=_style_params_from_cfg(doc_cfg or {})
    style_list = [
        ("BACKGROUND", (0,0), (-1,0), PRIMARY),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,0), 11),
        ("GRID", (0,0), (-1,-2), 0.5, BORDER),
        ("ROWBACKGROUNDS", (0,1), (-1,-3), [colors.white, LIGHT_BG]),
        ("LEFTPADDING", (0,0), (-1,-1), max(4, p["pad"]-2)),
        ("RIGHTPADDING", (0,0), (-1,-1), max(4, p["pad"]-2)),
        ("TOPPADDING", (0,0), (-1,-1), p["pad"]),
        ("BOTTOMPADDING", (0,0), (-1,-1), p["pad"]),
        ("ALIGN", (1,0), (-1,-1), "CENTER"),
        ("ALIGN", (0,0), (0,-1), "LEFT"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("FONTSIZE", (0,1), (-1,-3), 9),
    ]
    
    # Style total rows
    total_start = len(table_data) - 2
    style_list.extend([
        ("BACKGROUND", (0,total_start), (-1,-1), LIGHT_BG),
        ("FONTNAME", (0,total_start), (-1,-1), "Helvetica-Bold"),
        ("FONTSIZE", (0,total_start), (-1,-1), 10),
        ("TEXTCOLOR", (3,total_start), (3,-1), PRIMARY),
        ("SPAN", (0,total_start), (2,total_start)),
        ("SPAN", (0,total_start+1), (2,total_start+1)),
        ("GRID", (0,total_start), (-1,-1), 0.5, BORDER),
    ])
    
    # Highlight final total
    style_list.extend([
        ("BACKGROUND", (0,-1), (-1,-1), SUCCESS),
        ("TEXTCOLOR", (0,-1), (-1,-1), colors.white),
        ("FONTSIZE", (0,-1), (-1,-1), 11),
    ])
    
    if p["row_striping"]:
        style_list.append(("ROWBACKGROUNDS", (0,1), (-1,-3), [colors.white, LIGHT_BG]))
    tbl.setStyle(TableStyle(style_list))
    # Wrap table in KeepTogether to prevent splitting across pages
    elements.append(KeepTogether([tbl, Spacer(1, 8*mm)]))
    
    # Add explanation section
    elements.append(Paragraph("About Phone Services", styles["H2"]))
    elements.append(Spacer(1, 2*mm))
    
    explanation = [
        "Cloud-based phone system with mobile and desktop apps",
        "Services are quoted per phone user, not per total user",
        "Includes voicemail, call forwarding, and basic routing features",
        "Supplied and managed by Maxotel",
        "Optional add-ons available: call recording, advanced analytics, conference bridging"
    ]
    
    for point in explanation:
        elements.append(Paragraph(BUL + escape_html(point), styles["Body"]))
    
    elements.append(PageBreak())
    return elements

# ---------------------------
# NEW: Microsoft 365 Licensing Breakdown Renderer
# ---------------------------
def render_m365_licensing_section(m365_licenses: list[dict], gst_rate: float, styles, doc_cfg: dict | None = None) -> List[Any]:
    """
    Render Microsoft 365 licensing breakdown as a table.
    Returns list of flowables to insert into the quote page.
    """
    if not m365_licenses:
        return []
    
    elements = []
    
    # Title
    elements.append(Spacer(1, 6*mm))
    elements.append(Paragraph("Microsoft 365 Licensing", styles["H2"]))
    elements.append(Spacer(1, 2*mm))
    
    # Build table data
    table_data = [["License Type", "Quantity", "Per User", "Monthly Total"]]
    
    total_monthly = 0.0
    for lic in m365_licenses:
        item = lic["item"]
        qty = lic["quantity"]
        unit_price = lic["unit_price"]
        total = lic["total"]
        unit_type = lic.get("unit_type", "User")
        
        # Add unit type suffix if it's a service account
        display_name = item
        if unit_type == "Service Account":
            display_name = f"{item} (Service Account)"
        
        table_data.append([
            display_name,
            str(qty),
            f"${unit_price:,.2f}",
            f"${total:,.2f}"
        ])
        total_monthly += total
    
    # Add total row
    table_data.append(["", "", "", ""])  # Spacer
    table_data.append([
        "TOTAL M365 Licensing (ex GST)",
        "",
        "",
        f"${total_monthly:,.2f}"
    ])
    
    total_inc_gst = gst_inclusive(total_monthly, gst_rate)
    table_data.append([
        "TOTAL M365 Licensing (inc GST)",
        "",
        "",
        f"${total_inc_gst:,.2f}"
    ])
    
    # Create table
    col_widths = [80*mm, 25*mm, 25*mm, 30*mm]
    tbl = Table(table_data, colWidths=col_widths)
    
    p=_style_params_from_cfg(doc_cfg or {})
    style_list = [
        ("BACKGROUND", (0,0), (-1,0), PRIMARY),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,0), 11),
        ("GRID", (0,0), (-1,-2), 0.5, BORDER),
        ("ROWBACKGROUNDS", (0,1), (-1,-3), [colors.white, LIGHT_BG]),
        ("LEFTPADDING", (0,0), (-1,-1), max(4, p["pad"]-2)),
        ("RIGHTPADDING", (0,0), (-1,-1), max(4, p["pad"]-2)),
        ("TOPPADDING", (0,0), (-1,-1), p["pad"]),
        ("BOTTOMPADDING", (0,0), (-1,-1), p["pad"]),
        ("ALIGN", (1,0), (-1,-1), "CENTER"),
        ("ALIGN", (0,0), (0,-1), "LEFT"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("FONTSIZE", (0,1), (-1,-3), 9),
    ]
    
    # Style total rows
    total_start = len(table_data) - 2
    style_list.extend([
        ("BACKGROUND", (0,total_start), (-1,-1), LIGHT_BG),
        ("FONTNAME", (0,total_start), (-1,-1), "Helvetica-Bold"),
        ("FONTSIZE", (0,total_start), (-1,-1), 10),
        ("TEXTCOLOR", (3,total_start), (3,-1), PRIMARY),
        ("SPAN", (0,total_start), (2,total_start)),
        ("SPAN", (0,total_start+1), (2,total_start+1)),
        ("GRID", (0,total_start), (-1,-1), 0.5, BORDER),
    ])
    
    # Highlight final total
    style_list.extend([
        ("BACKGROUND", (0,-1), (-1,-1), SUCCESS),
        ("TEXTCOLOR", (0,-1), (-1,-1), colors.white),
        ("FONTSIZE", (0,-1), (-1,-1), 11),
    ])
    
    if p["row_striping"]:
        style_list.append(("ROWBACKGROUNDS", (0,1), (-1,-3), [colors.white, LIGHT_BG]))
    tbl.setStyle(TableStyle(style_list))
    
    elements.append(tbl)
    elements.append(Spacer(1, 6*mm))
    
    return elements

def draw_fullpage_bg(canv, bg_path: str | None):
    if not bg_path or not os.path.exists(bg_path):
        return
    try:
        pw, ph = canv._pagesize
        rdr = ImageReader(bg_path)
        iw, ih = rdr.getSize()
        scale = max(pw/iw, ph/ih)
        w, h = iw*scale, ih*scale
        x, y = (pw - w)/2.0, (ph - h)/2.0
        canv.saveState()
        canv.drawImage(rdr, x, y, width=w, height=h, mask="auto")
        canv.restoreState()
    except Exception as e:
        st.warning(f"Failed to draw background: {str(e)}")

def page_number(canv):
    pw, _ = canv._pagesize
    canv.setFont("Helvetica", 9)
    canv.setFillColor(MUTED)
    canv.drawRightString(pw - 18*mm, 18*mm, f"Page {canv._pageNumber}")
    canv.setStrokeColor(BORDER)
    canv.setLineWidth(0.5)
    canv.line(20*mm, 21*mm, pw - 20*mm, 21*mm)

def prepared_by(canv):
    canv.setFillColor(MUTED)
    canv.setFont("Helvetica", 9)
    canv.drawString(20*mm, 18*mm, "Prepared by Freedom IT")
    canv.setFont("Helvetica", 8)
    canv.drawString(20*mm, 15*mm, datetime.now().strftime("%B %d, %Y"))

def draw_footer_slogan(canv, slogan_path: str | None):
    """Draw the Freedom IT slogan image at the bottom of the page."""
    if not slogan_path or not os.path.exists(slogan_path):
        return
    try:
        pw, _ = canv._pagesize
        rdr = ImageReader(slogan_path)
        iw, ih = rdr.getSize()
        
        # Scale to fit width (max 100mm wide)
        max_width = 100*mm
        scale = min(max_width / iw, 1.0)
        w, h = iw*scale, ih*scale
        
        # Center horizontally, position above page number
        x = (pw - w) / 2.0
        y = 13*mm  # Just above the page number area
        
        canv.saveState()
        canv.drawImage(rdr, x, y, width=w, height=h, mask="auto")
        canv.restoreState()
    except Exception as e:
        st.warning(f"Failed to draw footer slogan: {str(e)}")

def gst_inclusive(ex: float, rate: float = 0.10) -> float:
    return round(float(ex) * (1 + float(rate)), 2)

def combined_services_support_page(q: Dict[str, Any], monthly_support_ex: float, gst_rate: float, support_points: List[str]) -> List[Any]:
    parts: List[Any] = []
    title = q.get("name", "Option")
    ex = float(q.get("ex_total", 0.0))
    desc_md = q.get("desc", "")
    services = q.get("services_included", [])

    parts.append(Paragraph(escape_html(title) + " - Combined", styles["H1"]))

    ex_inc = gst_inclusive(ex, gst_rate)
    sup_inc = gst_inclusive(monthly_support_ex, gst_rate)
    parts.append(Paragraph(
        f"<b>Monthly Services:</b> ${ex:,.2f} ex GST &nbsp;&nbsp;(${ex_inc:,.2f} inc)"
        f"<br/><b>Monthly Support:</b> ${monthly_support_ex:,.2f} ex GST &nbsp;&nbsp;(${sup_inc:,.2f} inc)",
        styles["Price"]
    ))
    parts.append(Spacer(1, 4))

    services_col: List[Any] = []
    for p in md_to_flowables(desc_md):
        services_col.append(p)
    services_col.append(Spacer(1, 3))
    services_col.append(Paragraph("Services Included", styles["H2"]))
    services_col.append(make_services_table(services, doc_cfg=doc_cfg))

    support_col: List[Any] = []
    support_col.append(Paragraph("Support - What's included", styles["H2"]))
    points = support_points or ["Unlimited remote support during agreed hours", "Proactive monitoring & patching", "Helpdesk, escalation & vendor liaison"]
    for pt in points[:MAX_INCLUDED_POINTS]:
        support_col.append(Paragraph(BUL + escape_html(pt), styles["Body"]))
    support_col.append(Spacer(1, 4))
    support_col.append(Paragraph("Support - Not included", styles["H2"]))
    for ni in ["On-site support (billed separately)", "Project work quoted separately", "Hardware & third-party licensing"]:
        support_col.append(Paragraph(BUL + escape_html(ni), styles["Body"]))

    too_long = len(desc_md) > MAX_DESC_LENGTH_FOR_TWO_COL or len(services) > MAX_SERVICES_FOR_TWO_COL or len(points) > MAX_POINTS_FOR_TWO_COL
    
    if too_long:
        parts.append(Paragraph("Services", styles["H2"]))
        parts.extend(services_col)
        parts.append(Spacer(1, 6))
        parts.append(Paragraph("Support", styles["H2"]))
        parts.extend(support_col)
        parts.append(PageBreak())
        return parts

    col_tbl = Table([[services_col, support_col]], colWidths=[80*mm, 75*mm])
    col_tbl.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 2),
        ("BOTTOMPADDING", (0,0), (-1,-1), 2),
    ]))
    parts.append(col_tbl)
    parts.append(PageBreak())
    return parts

    templates_dir = r"C:\FIT-Quoter\templates"

from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import Table, TableStyle

def section_banner(text, color="#0018A8", width=180*mm):
    # Use Body style for safer rendering across themes
    bar = Table([[Paragraph(f"<font color='white'><b>{text}</b></font>", styles["Body"])]],
                colWidths=[width])
    bar.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor(color)),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    return bar

def logo_wall(dir_path, cols=4, row_height=18*mm, width_mm=36):
    import os
    from reportlab.platypus import Image, Spacer, Table, TableStyle
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    # validate folder
    if not dir_path or not os.path.isdir(dir_path):
        return []
    files = [f for f in sorted(os.listdir(dir_path)) if f.lower().endswith((".png",".jpg",".jpeg",".gif",".bmp"))]
    if not files:
        return []
    # build images
    imgs = []
    for f in files:
        p = os.path.join(dir_path, f)
        try:
            imgs.append(Image(p, width=width_mm*mm, height=row_height, kind='proportional'))
        except Exception:
            continue
    if not imgs:
        return []
    # layout grid
    rows = [imgs[i:i+cols] for i in range(0, len(imgs), cols)]
    # pad last row to exact column count
    rem = len(rows[-1]) % cols
    if rem:
        rows[-1] += [Spacer(0, row_height)] * (cols - rem)
    # choose safe column width (cap to page width ~180mm)
    max_content_width = 180*mm
    col_w = min((width_mm+6)*mm, max_content_width/cols)
    grid = Table(rows, colWidths=[col_w]*cols, hAlign='LEFT')
    grid.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('BACKGROUND', (0,0), (-1,-1), colors.white),
    ]))
    return [grid]

# ==== Multi-LOGO wall helpers ====
import re as _logo_re, os as _logo_os, glob as _logo_glob
from reportlab.lib import colors as _logo_colors
from reportlab.lib.units import mm as _logo_mm
from reportlab.platypus import Paragraph as _logo_Paragraph, Table as _logo_Table, TableStyle as _logo_TableStyle, Image as _logo_Image, Spacer as _logo_Spacer
from reportlab.lib.styles import ParagraphStyle as _logo_ParaStyle
from reportlab.lib.units import mm  # if not already imported
from reportlab.platypus import Spacer, Paragraph, Table, TableStyle, Image  # if not already imported

_LOGO_BLOCK_RE = _logo_re.compile(
    r"<!--\s*LOGO_HEADING=(?P<heading>.*?)\s*-->\s*"
    r"<!--\s*LOGO_DIR=(?P<dir>.*?)\s*-->\s*"
    r"<!--\s*LOGO_COLS=(?P<cols>\d+)\s*-->\s*"
    r"<!--\s*LOGO_SIZE_MM=(?P<size>\d+)\s*-->",
    _logo_re.IGNORECASE | _logo_re.DOTALL
)

def extract_logo_blocks(md_text: str):
    blocks = []
    def _sub(m):
        idx = len(blocks)
        blocks.append({
            "heading": m.group("heading").strip(),
            "logo_dir": m.group("dir").strip(),
            "cols": int(m.group("cols")),
            "size_mm": int(m.group("size")),
        })
        # Wrap in <p> tags so htmlish_to_flowables will capture it
        return f"\n\n<p>[[LOGO_BLOCK_{idx}]]</p>\n\n"
    cleaned = _LOGO_BLOCK_RE.sub(_sub, md_text)
    return cleaned, blocks

def _render_logo_wall(heading: str, logo_dir: str, cols: int, size_mm_val: int, styles):
    flow = []
    bar_style = _logo_ParaStyle(
        "LogoBar", parent=styles["Normal"], textColor=_logo_colors.white,
        backColor=_logo_colors.HexColor("#5B9BD5"), fontSize=10, leading=12,
        leftIndent=4, spaceBefore=8, spaceAfter=6,
    )
    flow.append(_logo_Paragraph(f"<b>{heading}</b>", bar_style))
    flow.append(_logo_Spacer(1, 4))

    files = []
    if _logo_os.path.isdir(logo_dir):
        for ext in ("png", "jpg", "jpeg", "gif"):
            files.extend(sorted(_logo_glob.glob(_logo_os.path.join(logo_dir, f"*.{ext}"))))

    if not files:
        warn = _logo_ParaStyle("Warn", parent=styles["Normal"], textColor=_logo_colors.HexColor("#FF0000"), fontSize=8)
        flow.append(_logo_Paragraph(f"(No images found in {logo_dir})", warn))
        flow.append(_logo_Spacer(1, 6))
        return flow

    size = size_mm_val * _logo_mm
    cells, row = [], []
    for f in files:
        try:
            img = _logo_Image(f, width=size, height=size, kind="proportional")
        except Exception:
            continue
        inner = _logo_Table([[img]], colWidths=[size], rowHeights=[size])
        inner.setStyle(_logo_TableStyle([
            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ]))
        row.append(inner)
        if len(row) == cols:
            cells.append(row); row = []
    if row:
        while len(row) < cols:
            row.append(_logo_Spacer(0, size))
        cells.append(row)

    table = _logo_Table(cells, hAlign="LEFT")
    table.setStyle(_logo_TableStyle([
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    flow.append(table)
    flow.append(_logo_Spacer(1, 8))
    return flow

def _expand_logo_placeholder(flowable, blocks, styles):
    try:
        txt = flowable.getPlainText().strip()
    except (AttributeError, Exception):
        # Not a Paragraph or doesn't have getPlainText
        return None
    
    # Remove any residual HTML tags that might be in the text
    txt = txt.replace("<p>", "").replace("</p>", "").strip()
    txt = _logo_re.sub(r'<[^>]+>', '', txt).strip()  # Remove any other HTML tags
    
    if not txt:
        return None
    
    # Check if this is a logo block placeholder
    if not (txt.startswith("[[LOGO_BLOCK_") and txt.endswith("]]")):
        return None
    
    m = _logo_re.search(r"\[\[LOGO_BLOCK_(\d+)\]\]", txt)
    if not m:
        return None
    idx = int(m.group(1))
    if not (0 <= idx < len(blocks)):
        return []
    blk = blocks[idx]
    return _render_logo_wall(
        heading=blk["heading"], logo_dir=blk["logo_dir"],
        cols=blk["cols"], size_mm_val=blk["size_mm"], styles=styles
    )
# ==== /Multi-LOGO wall helpers ====


# === Helpers for 'Services Included' section ===
from reportlab.platypus import Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.units import mm

# --------------------------------------------------------------------------------------
# Services Included Utilities
# --------------------------------------------------------------------------------------

def services_from_quote(quote_or_items) -> list[str]:
    """Return a clean, de-duplicated list of included services.
    Accepts either a quote dict (with .get('items') / .get('lines')) OR a list of item dicts.
    """
    if not quote_or_items:
        return []

    # Normalize to a list of item dicts
    if isinstance(quote_or_items, list):
        items = quote_or_items
    elif isinstance(quote_or_items, dict):
        items = quote_or_items.get("items") or quote_or_items.get("lines") or []
    else:
        items = []
    
    # FIX: Flatten if we accidentally got nested lists
    # Some quote formats may have items wrapped in additional lists
    if items and isinstance(items, list):
        flattened = []
        for item in items:
            if isinstance(item, list):
                # It's a nested list - flatten it
                flattened.extend([i for i in item if isinstance(i, dict)])
            elif isinstance(item, dict):
                flattened.append(item)
        items = flattened

    names = []
    EXCLUDE = ("onboarding", "hardware", "project", "one-off", "once-off", "gst", "shipping")

    for item in items:
        # FIX: Ensure item is a dictionary before calling .get()
        if not isinstance(item, dict):
            continue

        # FIX: Use try-except to safely extract name even if item structure is unexpected
        try:
            name = (item.get("name") or item.get("description") or "").strip()
        except (AttributeError, TypeError):
            continue
            
        if not name:
            continue

        lower = name.lower()
        if any(x in lower for x in EXCLUDE):
            continue

        # skip pure notes or zero-priced rows
        try:
            unit_price = float(item.get("unit_price", 0) or item.get("price", 0) or 0)
            total_price = float(item.get("total", 0) or 0)
        except (AttributeError, TypeError, ValueError):
            unit_price = 0.0
            total_price = 0.0

        if unit_price == 0 and total_price == 0:
            continue

        names.append(name)

    # Friendly renames for cleaner presentation
    RENAMES = {
        "EDR": "Managed Threat Detection & Response",
        "SentinelOne": "Managed Threat Detection & Response",
        "M365 Backup": "Microsoft 365 Backup & Archiving",
        "Microsoft 365 Business Standard": "Microsoft 365 Productivity",
        "Freedom IT Agent": "Freedom IT Support Agent",
    }

    mapped = [RENAMES.get(n, n) for n in names]

    # De-duplicate while preserving order
    seen, result = set(), []
    for n in mapped:
        if n not in seen:
            result.append(n)
            seen.add(n)

    return result



def render_services_included(services: list[str], styles, cols: int = 2, doc_cfg: dict | None = None):
    """Render a two-column bullet list titled 'Services Included'."""
    if not services:
        return []

    out = [Paragraph("<b>Services Included</b>", styles.get("H2", styles.get("Heading2")))]

    body_style = styles.get("Body", None) or styles.get("Normal")
    bullets = [Paragraph(f"• {s}", body_style) for s in services]

    # Split into columns
    per_col = (len(bullets) + cols - 1) // cols
    cols_data = [bullets[i * per_col:(i + 1) * per_col] for i in range(cols)]
    max_len = max((len(c) for c in cols_data), default=0)

    for c in cols_data:
        while len(c) < max_len:
            c.append(Paragraph("&nbsp;", body_style))

    rows = list(zip(*cols_data)) if cols_data else []
    table = Table(rows, colWidths=[90 * mm] * cols, hAlign="LEFT")

    # Get density settings
    p = _style_params_from_cfg(doc_cfg or {})

    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), max(8, p["pad"]+4)),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), max(2, p["pad"]-3)),
    ]))

    out.append(Spacer(1, 4 * mm))
    out.append(table)
    out.append(Spacer(1, 6 * mm))
    return out

def build_pdf(
    client_name: str, 
    cover_md: str, 
    cover_overview_img: bytes | None, 
    client_logo: bytes | None, 
    freedom_logo_path: str | None, 
    quotes: List[Dict[str, Any]], 
    cover_bg_path: str | None, 
    inner_bg_path: str | None, 
    snapshot: dict | None, 
    monthly_support_ex: float, 
    gst_rate: float, 
    included_points: List[str], 
    combine_support_and_services: bool,
    site_services: list[dict] | None = None,
    phone_services: list[dict] | None = None,
    footer_slogan_path: str | None = None,
    site_desc_md: str = "",
    phone_desc_md: str = "",
    onboarding_monthlyised: float = 0.0,
    onboarding_outright: float = 0.0,
    doc_cfg: dict | None = None,
    nerdio_estimate: Optional[Any] = None
) -> bytes:
    # DEFENSIVE: Ensure doc_cfg is always a dict or None
    # Sometimes it's passed as a list by accident, which causes AttributeError
    if not isinstance(doc_cfg, (dict, type(None))):
        doc_cfg = {}  # Normalize to empty dict if it's wrong type
    
    buf = io.BytesIO()
    layout = (doc_cfg or {}).get("layout", {})
    m = layout.get("page_margins_mm", 18)
    try:
        if isinstance(m, dict):
            left = float(m.get("left", 18))*mm
            right = float(m.get("right", 18))*mm
            top = float(m.get("top", 18))*mm
            bottom = float(m.get("bottom", 18))*mm
        else:
            left = right = top = bottom = float(m)*mm
    except Exception:
        left = right = top = bottom = 18*mm

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=left, rightMargin=right,
        topMargin=top, bottomMargin=bottom
    )
    elements: List[Any] = []

    # --- Header logos ---
    left_logo = safe_image_flowable(client_logo, max_w=48*mm, max_h=22*mm)
    right_logo = safe_image_flowable(freedom_logo_path, max_w=48*mm, max_h=22*mm)
    t = Table([[left_logo, right_logo]], colWidths=[doc.width/2, doc.width/2])
    t.setStyle(TableStyle([
        ("ALIGN", (0,0), (0,0), "LEFT"),
        ("ALIGN", (1,0), (1,0), "RIGHT"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 6*mm))

    elements.append(Paragraph("IT Services Proposal", styles["TitleCentered"]))
    if client_name.strip():
        elements.append(Spacer(1, 2*mm))
        elements.append(Paragraph(
            f"Prepared for <font color='{PRIMARY}'><b>{escape_html(client_name)}</b></font>",
            styles["CoverClient"]
        ))
    elements.append(Spacer(1, 8*mm))

    # --- Divider ---
    line_table = Table([[""]], colWidths=[doc.width])
    line_table.setStyle(TableStyle([
        ("LINEABOVE", (0,0), (-1,0), 2, PRIMARY),
        ("TOPPADDING", (0,0), (-1,-1), 0),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    elements.append(line_table)
    elements.append(Spacer(1, 4*mm))

    # --- Cover text & overview image layout logic ---
    cover_flowables = md_to_flowables(cover_md)
    estimated_text_height = len(cover_flowables) * 12
    available_space = 40 if cover_overview_img else 160

    if estimated_text_height > available_space and cover_overview_img:
        for p in cover_flowables:
            elements.append(p)
        elements.append(PageBreak())

        elements.append(Paragraph("Proposal Overview", styles["H1"]))
        elements.append(Spacer(1, 4*mm))
        elements.append(safe_image_flowable(cover_overview_img, A4[0]-40*mm, 120*mm))
        elements.append(Spacer(1, 4*mm))
        elements.append(PageBreak())
    else:
        for p in cover_flowables:
            elements.append(p)
        elements.append(Spacer(1, 6*mm))
        if cover_overview_img:
            elements.append(safe_image_flowable(cover_overview_img, A4[0]-40*mm, 120*mm))
            elements.append(Spacer(1, 4*mm))
        elements.append(PageBreak())

    # --- Static Marketing Page ---
    try:
        templates_dir_var = r"C:\\FIT-Quoter\\templates"
        _title, _mkt_md = load_template_markdown("marketing_static", templates_dir_var)

        _mkt_body, _logo_blocks = extract_logo_blocks(_mkt_md)
        _base_flow = htmlish_to_flowables(_mkt_body) if "<!-- FORMAT:HTML -->" in _mkt_md else md_to_flowables(_mkt_body)
        _expanded = []
        for f in _base_flow:
            repl = _expand_logo_placeholder(f, _logo_blocks, styles)
            if repl is None:
                _expanded.append(f)
            else:
                _expanded.extend(repl)
        elements.extend(_expanded)
        elements.append(PageBreak())
    except Exception as _mkt_err:
        st.warning(f"Static Marketing Page failed: {_mkt_err}")
        _fallback = ('<h1><font color="#0018A8">We become your technology partner</font></h1>'
                     '<p>We align your technology with your business goals — simplifying support, '
                     'lifting security, and freeing your team to focus on what matters.</p>')
        elements.extend(htmlish_to_flowables(_fallback))
        elements.append(PageBreak())

    # --- Client Overview + Onboarding ---
    if snapshot:
        elements.append(Paragraph("Client Overview", styles["H1"]))
        elements.append(Paragraph("This proposal has been tailored specifically for your organization's needs.", styles["SectionIntro"]))
        elements.append(Spacer(1, 4*mm))
        elements.append(make_snapshot_table(snapshot))

        _onb_mode = ((doc_cfg or {}).get("pricing", {}) or {}).get("onboarding_mode", "Show Both")
        _show_outright = _onb_mode in ("Show Both", "Show Outright only")
        _show_monthly = _onb_mode in ("Show Both", "Show Monthlyised only")

        if ((_show_monthly and onboarding_monthlyised and onboarding_monthlyised > 0) or
            (_show_outright and onboarding_outright and onboarding_outright > 0)):
            elements.append(Spacer(1, 6*mm))
            elements.append(Paragraph("Onboarding Fee", styles["H2"]))
            ob_rows = [["Type", "Amount"]]
            if _show_outright and onboarding_outright and onboarding_outright > 0:
                ob_rows.append(["One-off (Outright)", f"${onboarding_outright:,.2f} ex GST"])
                if (doc_cfg or {}).get("pricing",{}).get("show_inc", True):
                    ob_rows.append(["", f"${gst_inclusive(onboarding_outright, gst_rate):,.2f} inc GST"])
            if _show_monthly and onboarding_monthlyised and onboarding_monthlyised > 0:
                ob_rows.append(["Monthlyised (12 months)", f"${onboarding_monthlyised:,.2f} ex GST"])
                if (doc_cfg or {}).get("pricing",{}).get("show_inc", True):
                    ob_rows.append(["", f"${gst_inclusive(onboarding_monthlyised, gst_rate):,.2f} inc GST"])

            ob_tbl = Table(ob_rows, colWidths=[70*mm, 90*mm])
            ob_tbl.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), PRIMARY),
                ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                ("GRID", (0,0), (-1,-1), 0.5, BORDER),
                ("LEFTPADDING", (0,0), (-1,-1), 8),
                ("RIGHTPADDING", (0,0), (-1,-1), 8),
                ("TOPPADDING", (0,0), (-1,-1), 6),
                ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ]))
            elements.append(ob_tbl)
            elements.append(Spacer(1, 4*mm))

            elements.append(Paragraph("About Onboarding", styles["H2"]))
            elements.append(Spacer(1, 2*mm))
            for point in [
                "One-time fee to onboard your business to Freedom IT's systems and processes",
                "Outright: Paid upfront as a single payment",
                "Monthlyised: Spread evenly across 12 monthly payments for easier budgeting",
                "Covers initial setup, documentation, system integration, and team training",
            ]:
                elements.append(Paragraph(BUL + escape_html(point), styles["Body"]))
            elements.append(Spacer(1, 4*mm))
        elements.append(PageBreak())
    
    # --- Site & Phone pages ---
    if site_services and snapshot and snapshot.get("sites"):
        num_sites = snapshot.get("sites")
        elements.extend(site_infrastructure_page(site_services, num_sites, gst_rate, site_desc_md, doc_cfg=doc_cfg))

    if phone_services and snapshot and snapshot.get("phone_users"):
        num_phone_users = snapshot.get("phone_users")
        elements.extend(phone_services_page(phone_services, num_phone_users, gst_rate, phone_desc_md, doc_cfg=doc_cfg))

    # --- Multi-Quote Comparison ---
    if len(quotes) > 1 and _include_section(doc_cfg, "Plan Comparison"):
        elements.append(Paragraph("Plan & Pricing Comparison", styles["H1"]))
        elements.append(Paragraph("Compare features, services, and investment across all available options.", styles["SectionIntro"]))
        elements.append(Spacer(1, 4*mm))
        comparison_elements = make_comprehensive_comparison_table(quotes, gst_rate, monthly_support_ex, doc_cfg=doc_cfg)
        elements.extend(comparison_elements)
        elements.append(Spacer(1, 6*mm))
        site_ex_total = sum_monthly_total(site_services)
        phone_ex_total = sum_monthly_total(phone_services)
        if _include_section(doc_cfg, "Monthly Investment Breakdown"):
            elements.extend(make_monthly_totals_by_quote(quotes, site_ex_total, phone_ex_total, gst_rate, doc_cfg=doc_cfg))
        elements.append(PageBreak())

    # --- Per-Quote Pages ---
    has_services = len(quotes) > 0
    has_support = any(float(q.get("support_ex", 0.0) or 0.0) > 0 for q in quotes)
    if monthly_support_ex and float(monthly_support_ex) > 0:
        has_support = True

    _combine_sidebar = (doc_cfg or {}).get("options",{}).get("combine_support_and_services", combine_support_and_services)
    if has_services and has_support and _combine_sidebar:
        q = quotes[0]
        elements.extend(combined_services_support_page(q, monthly_support_ex, gst_rate, included_points))
    else:
        for idx, q in enumerate(quotes):
            if idx > 0:
                elements.append(PageBreak())

            title = q.get("name", "Option")
            ex = float(q.get("ex_total", 0.0))
            inc = gst_inclusive(ex, gst_rate)
            desc_md = q.get("desc", "")
            opt_img = q.get("img_bytes")

            elements.append(Paragraph(escape_html(title), styles["H1"]))
            title_line = Table([[""]], colWidths=[160*mm])
            title_line.setStyle(TableStyle([
                ("LINEBELOW", (0,0), (-1,0), 2, PRIMARY),
                ("TOPPADDING", (0,0), (-1,-1), 2),
                ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ]))
            elements.append(title_line)
            elements.append(Spacer(1, 2*mm))
            
            # Render quote description markdown if present
            if desc_md and desc_md.strip():
                try:
                    desc_flowables = md_to_flowables(desc_md)
                    elements.extend(desc_flowables)
                    elements.append(Spacer(1, 4*mm))
                except Exception as desc_err:
                    # If markdown rendering fails, just skip it
                    pass

            # Determine display type and additional pricing info
            display_type = "monthly"  # default
            hourly_rate = None
            support_amount = None
            
            # FIXED: Use per-quote data if available, fallback to shared snapshot
            # This allows each quote to have its own tier/rate settings
            rate_type = str(q.get("rate_display_type") or snapshot.get("rate_display_type", "monthly")).strip().lower()
            hourly_rate_value = q.get("hourly_rate") or snapshot.get("hourly_rate")
            tier = str(q.get("proposal_type") or snapshot.get("proposal_type", "")).lower()
            quote_support_ex = float(q.get("support_ex", 0.0) or 0.0)
            
            # Primary logic: Use rate_display_type if available and reliable
            # But also check tier as a fallback for managed quotes
            is_adhoc = ("adhoc" in tier or "ad hoc" in tier or rate_type == "hourly")
            is_managed = ("managed" in tier or rate_type == "managed")
            
            # Priority 1: ADHOC with hourly rate
            # Show hourly rate if:
            # - Rate type is hourly OR tier is ad-hoc
            # - Hourly rate value exists and > 0
            # - NOT managed
            # - Support is 0 or not set (critical: ad-hoc quotes should not have support)
            if is_adhoc and hourly_rate_value is not None and hourly_rate_value > 0 and not is_managed:
                # Only show hourly if support is actually 0
                if quote_support_ex == 0 or quote_support_ex is None:
                    display_type = "hourly"
                    hourly_rate = hourly_rate_value
                    support_amount = None
                else:
                    # Has support despite being marked ad-hoc - treat as managed
                    display_type = "managed"
                    support_amount = quote_support_ex
                    hourly_rate = None
            # Priority 2: MANAGED with monthly support
            # Show monthly support if there's a support amount > 0
            elif quote_support_ex and quote_support_ex > 0:
                support_amount = quote_support_ex  # Use THIS quote's support amount
                display_type = "managed"
                hourly_rate = None  # Explicitly no hourly rate for managed
            # Priority 3: STANDARD - Just monthly services
            else:
                display_type = "monthly"
                hourly_rate = None
                support_amount = None
            
            # Create pricing box with appropriate display
            elements.append(make_price_box(
                "Monthly Services", 
                ex, 
                inc, 
                gst_rate,
                display_type=display_type,
                hourly_rate=hourly_rate,
                support_amount=support_amount,
                doc_cfg=doc_cfg
            ))
            
            # --- Services Included section ---
            # Extract and render services from the quote if enabled
            # Check in options (not sections, since sections is a list)
            _show_services = True  # Default to showing services
            if isinstance(doc_cfg, dict):
                _show_services = doc_cfg.get("options", {}).get("services_included", True)
                _services_debug = doc_cfg.get("options", {}).get("services_debug", False)
            else:
                _services_debug = False
            
            if _show_services:
                # Try to extract services from this quote
                _src = None
                if isinstance(q, dict):
                    _items = q.get("items") or q.get("lines")
                    if _items:
                        _src = _items
                    else:
                        _src = q
                elif isinstance(q, list):
                    _src = q
                
                # DEBUG: Show what we're working with
                if _services_debug:
                    st.write("📁 **Services Debug Info:**")
                    st.write(f"- Quote type: {type(q)}")
                    st.write(f"- Has 'items': {('items' in q) if isinstance(q, dict) else 'N/A'}")
                    st.write(f"- Has 'lines': {('lines' in q) if isinstance(q, dict) else 'N/A'}")
                    if _src:
                        st.write(f"- Source type: {type(_src)}")
                        if isinstance(_src, list):
                            st.write(f"- Source length: {len(_src)}")
                            if len(_src) > 0:
                                st.write(f"- First item type: {type(_src[0])}")
                                st.write(f"- First item sample: {str(_src[0])[:200]}...")
                
                if _src:
                    _svc = services_from_quote(_src)
                    
                    # DEBUG: Show extracted services
                    if _services_debug:
                        st.write(f"- Extracted services: {len(_svc)}")
                        if _svc:
                            st.write("- Services list:")
                            for s in _svc:
                                st.write(f"  • {s}")
                    
                    if _svc:
                        elements.append(Spacer(1, 6*mm))
                        elements.extend(render_services_included(_svc, styles, cols=2, doc_cfg=doc_cfg))
                    elif _services_debug:
                        st.warning("⚠️ No services extracted from quote items")
                elif _services_debug:
                    st.warning("⚠️ No items/lines found in quote data")
                    # If no services found, don't show anything (cleaner)
            
            # --- NEW: Microsoft 365 Licensing Breakdown section ---
            # This shows detailed M365 licensing with quantities and pricing
            if 'm365_licenses' in globals() and m365_licenses:
                m365_section = render_m365_licensing_section(m365_licenses, gst_rate, styles, doc_cfg=doc_cfg)
                elements.extend(m365_section)


    # --- Monthly Support Explained section ---
    if has_support and _include_section(doc_cfg, "Support Overview"):
        elements.append(PageBreak())
        try:
            templates_dir_var = r"C:\\FIT-Quoter\\templates"
            _title, _md = load_template_markdown("support_explained", templates_dir_var)
            elements.extend(md_to_flowables(_md))
        except Exception as _supp_err:
            st.warning(f"Failed to render Support Overview: {_supp_err}")

    # --- Page Decorations ---
    def on_first(canv, _doc):
        draw_fullpage_bg(canv, cover_bg_path)
        prepared_by(canv)
        draw_footer_slogan(canv, footer_slogan_path)
        if (doc_cfg or {}).get('layout',{}).get('footer_note'):
            canv.setFont('Helvetica',8); canv.setFillColor(MUTED)
            canv.drawCentredString(canv._pagesize[0]/2.0, 13*mm, (doc_cfg or {}).get('layout',{}).get('footer_note'))
        if (doc_cfg or {}).get('layout',{}).get('show_page_numbers', True):
            page_number(canv)

    def on_later(canv, _doc):
        draw_fullpage_bg(canv, inner_bg_path)
        draw_footer_slogan(canv, footer_slogan_path)
        if (doc_cfg or {}).get('layout',{}).get('footer_note'):
            canv.setFont('Helvetica',8); canv.setFillColor(MUTED)
            canv.drawCentredString(canv._pagesize[0]/2.0, 13*mm, (doc_cfg or {}).get('layout',{}).get('footer_note'))
        if (doc_cfg or {}).get('layout',{}).get('show_page_numbers', True):
            page_number(canv)

    # --- Render Dynamic Templates ---
    # Render any dynamic markdown templates that are in the sections list
    if isinstance(doc_cfg, dict):
        sections_list = doc_cfg.get("sections", [])
        dynamic_templates_dict = doc_cfg.get("dynamic_templates", {})
        templates_dir_var = r"C:\\FIT-Quoter\\templates"
        
        for section_name in sections_list:
            # Check if this section is a dynamic template
            if section_name in dynamic_templates_dict:
                template_stem = dynamic_templates_dict[section_name]
                try:
                    elements.append(PageBreak())
                    _title, _template_md = load_template_markdown(template_stem, templates_dir_var)
                    
                    # FIXED: Check if template is HTML format and use appropriate renderer
                    if "<!-- FORMAT:HTML -->" in _template_md:
                        # Strip the format marker
                        _template_body = _template_md.replace("<!-- FORMAT:HTML -->", "").strip()
                        
                        # Extract logo blocks before processing HTML
                        _cleaned, _logo_blocks = extract_logo_blocks(_template_body)
                        
                        # Process HTML
                        _base_flow = htmlish_to_flowables(_cleaned)
                        
                        # Expand logo placeholders
                        for f in _base_flow:
                            expanded = _expand_logo_placeholder(f, _logo_blocks, styles)
                            if expanded:
                                elements.extend(expanded)
                            else:
                                elements.append(f)
                    else:
                        # Standard markdown format
                        elements.extend(md_to_flowables(_template_md))
                except Exception as _dyn_err:
                    # Log error but continue
                    st.warning(f"Failed to render dynamic template '{section_name}': {_dyn_err}")

    # === V6.0 NEW: Add Nerdio estimate page if provided ===
    if nerdio_estimate:
        from reportlab.lib import colors
        
        elements.append(PageBreak())
        
        # Title
        elements.append(Paragraph("Azure Virtual Desktop Cost Estimate", styles["H1"]))
        elements.append(Spacer(1, 0.3 * inch))
        
        # Summary section
        elements.append(Paragraph("Summary", styles["H2"]))
        elements.append(Spacer(1, 0.1 * inch))
        
        # Get density settings for Nerdio tables
        p = _style_params_from_cfg(doc_cfg or {})
        
        summary_data = [
            [Paragraph("<b>Cost per User/Month</b>", styles["Body"]), nerdio_estimate.cost_per_user],
            [Paragraph("<b>Total Monthly Cost</b>", styles["Body"]), f"${nerdio_estimate.cost_per_month:,.2f}"],
            [Paragraph("<b>Number of Users</b>", styles["Body"]), str(nerdio_estimate.users_count)],
            [Paragraph("<b>Region</b>", styles["Body"]), nerdio_estimate.region]
        ]
        summary_table = Table(summary_data, colWidths=[3*inch, 3*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), max(8, p["pad"]+4)),
            ('RIGHTPADDING', (0, 0), (-1, -1), max(8, p["pad"]+4)),
            ('TOPPADDING', (0, 0), (-1, -1), p["pad"]),
            ('BOTTOMPADDING', (0, 0), (-1, -1), p["pad"]),
        ]))
        elements.append(summary_table)
        elements.append(Spacer(1, 0.3 * inch))
        
        # Cost breakdown
        elements.append(Paragraph("Cost Breakdown", styles["H2"]))
        elements.append(Spacer(1, 0.1 * inch))
        
        breakdown_data = [
            [Paragraph("<b>Component</b>", styles["Body"]), Paragraph("<b>Monthly Cost</b>", styles["Body"])],
            ["Azure Infrastructure", f"${nerdio_estimate.azure_cost:,.2f}"],
            ["Nerdio Manager", f"${nerdio_estimate.nerdio_cost:,.2f}"],
            ["Microsoft 365", f"${nerdio_estimate.m365_cost:,.2f}"],
            ["Margin", f"${nerdio_estimate.margin_total:,.2f}"]
        ]
        breakdown_table = Table(breakdown_data, colWidths=[3*inch, 3*inch])
        breakdown_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0078d4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), max(8, p["pad"]+4)),
            ('RIGHTPADDING', (0, 0), (-1, -1), max(8, p["pad"]+4)),
            ('TOPPADDING', (0, 0), (-1, -1), p["pad"]),
            ('BOTTOMPADDING', (0, 0), (-1, -1), p["pad"]),
        ]))
        elements.append(breakdown_table)
        
        # Desktop pools if available
        if nerdio_estimate.desktop_pools:
            elements.append(Spacer(1, 0.3 * inch))
            elements.append(Paragraph("Desktop Pools", styles["H2"]))
            elements.append(Spacer(1, 0.1 * inch))
            
            for pool in nerdio_estimate.desktop_pools:
                pool_title = Paragraph(f"<b>{pool['name']}</b>", styles["Body"])
                elements.append(pool_title)
                elements.append(Spacer(1, 0.05 * inch))
                
                pool_data = [
                    [Paragraph("<b>Users</b>", styles["Body"]), str(pool['users'])],
                    [Paragraph("<b>VM Size</b>", styles["Body"]), pool['vm_size']],
                    [Paragraph("<b>Specs</b>", styles["Body"]), f"{pool['cores']} cores, {pool['memory']} GB RAM"],
                    [Paragraph("<b>Autoscale</b>", styles["Body"]), f"{pool['min_desktops']}-{pool['max_desktops']} hosts"],
                    [Paragraph("<b>Hours</b>", styles["Body"]), f"{pool['min_hours']}h min, {pool['max_hours']}h peak"]
                ]
                pool_table = Table(pool_data, colWidths=[2*inch, 4*inch])
                pool_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f9f9f9')),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
                    ('LEFTPADDING', (0, 0), (-1, -1), max(6, p["pad"]+2)),
                    ('RIGHTPADDING', (0, 0), (-1, -1), max(6, p["pad"]+2)),
                    ('TOPPADDING', (0, 0), (-1, -1), p["pad"]),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), p["pad"]),
                ]))
                elements.append(pool_table)
                elements.append(Spacer(1, 0.15 * inch))
        
        # Licenses if available
        if nerdio_estimate.licenses:
            elements.append(Spacer(1, 0.2 * inch))
            elements.append(Paragraph("Licenses", styles["H2"]))
            elements.append(Spacer(1, 0.1 * inch))
            
            license_items = []
            for lic in nerdio_estimate.licenses:
                license_items.append(f"• <b>{lic['type']}:</b> {lic['plan']} × {lic['count']}")
            
            for item in license_items:
                elements.append(Paragraph(item, styles["Body"]))
                elements.append(Spacer(1, 0.05 * inch))
        
        # Autoscale savings if available
        if nerdio_estimate.autoscale.get('total', 0) > 0:
            elements.append(Spacer(1, 0.2 * inch))
            elements.append(Paragraph("Autoscale Savings", styles["H2"]))
            elements.append(Spacer(1, 0.1 * inch))
            savings_text = f"<b>Monthly Savings:</b> ${nerdio_estimate.autoscale['total']:,.2f}"
            elements.append(Paragraph(savings_text, styles["Body"]))

    doc.build(elements, onFirstPage=on_first, onLaterPages=on_later)
    return buf.getvalue()

st.set_page_config(page_title="Freedom IT - Quote Portal", page_icon="📁", layout="centered")
st.title("Freedom IT - Quote Portal (25-10-25)")
st.caption("Generate branded client proposals with Service Accounts, Site Infrastructure, and Phone Services support.")

BG_CHOICES = ["None", "Outside Border - Style 1.png", "Outside Border - Style 2.png", "Outside Border - Style 3.png"]

# Detect available Freedom IT logos
LOGO_FILES = []
possible_logos = [
    "freedom_it_logo.png",
    "Freedom IT Logo.png",
    "FreedomIT_Logo.png", 
    "Freedom_IT_Logo.png",
    "logo.png",
    "Logo.png"
]
for logo_file in possible_logos:
    if os.path.exists(logo_file):
        LOGO_FILES.append(logo_file)

if not LOGO_FILES:
    LOGO_FILES = ["freedom_it_logo.png"]  # Default fallback

# Footer slogan file
FOOTER_SLOGAN_FILE = "Turning IT Obstacles Into Business Stepping Stones.png"

def resolve_bg(choice: str) -> str | None:
    if choice == "None": 
        return None
    if choice not in BG_CHOICES[1:]:
        return None
    p = os.path.abspath(choice)
    return p if os.path.exists(p) else None

# FIXED: Initialize all variables BEFORE they are used in the UI
quotes_data: List[Dict[str, Any]] = []
first_snapshot: dict | None = None
merged_included_points: List[str] = []
site_services_list: list[dict] = []
phone_services_list: list[dict] = []
m365_licenses: list[dict] = []
monthly_support_ex_final: float = 0.0
site_desc_md: str = ""
phone_desc_md: str = ""
quote_overview_md: str = ""
onboarding_monthlyised_final: float = 0.0
onboarding_outright_final: float = 0.0

with st.expander("Client & Cover", expanded=True):
    c1, c2 = st.columns([1,1])
    client_name = c1.text_input("Client name", value="", key="client_name")
    client_logo_file = c2.file_uploader("Client logo (PNG/JPG)", type=["png","jpg","jpeg"], key="client_logo_file")

    overview_img_file = st.file_uploader("Optional cover overview image (PNG/JPG)", type=["png","jpg","jpeg"], key="overview_img_file")

    st.markdown("**Cover page description (Markdown)**")
    cover_md = st.text_area("Markdown supports #, ##, ###, bullets (- *  ), **bold**, *italic*.", height=200, placeholder="## Overview\nYour introduction here...\n\n- Key value point\n- Another point\n", label_visibility="collapsed", key="cover_md")
    # Prefill session_state once with extracted Quote Overview from first file
    if ('excel_files' in st.session_state and st.session_state['excel_files'] 
        and (not st.session_state.get('cover_md') or not st.session_state['cover_md'].strip()) 
        and quote_overview_md):
        st.session_state['cover_md'] = quote_overview_md
        cover_md = st.session_state['cover_md']

    st.markdown("**Live preview**")
    st.markdown(cover_md if cover_md.strip() else "_Nothing to preview yet._")
    
    # COVER TEXT OVERFLOW WARNING
    if cover_md.strip():
        # Rough estimation: count markdown elements
        lines = cover_md.split('\n')
        sections = len([l for l in lines if l.strip().startswith('#')])
        paragraphs = len([l for l in lines if l.strip() and not l.strip().startswith('#') and not l.strip().startswith('-')])
        bullets = len([l for l in lines if l.strip().startswith('-') or l.strip().startswith('*')])
        
        # Rough height estimate (mm): headers=15, paragraphs=12, bullets=6
        estimated_height = (sections * 15) + (paragraphs * 12) + (bullets * 6)
        
        # Check if we have a cover image
        has_cover_image = overview_img_file is not None
        available_space = 40 if has_cover_image else 160
        
        if estimated_height > available_space:
            if has_cover_image:
                st.info(f"""
                ℹ️ **Cover text is longer than usual** (~{estimated_height:.0f}mm content, {available_space}mm available with image)
                
                **The PDF will automatically:**
                - Keep your text on the cover page
                - Move the overview image to page 2
                
                This maintains a professional appearance without truncating your content.
                """)
            else:
                st.warning(f"""
                ⚠️ **Cover text is very long** (~{estimated_height:.0f}mm content, {available_space}mm available)
                
                **Recommendations:**
                1. Consider shortening by ~{int((estimated_height - available_space) / 12)} paragraphs
                2. Or let it flow to a second page (will add continuation heading)
                """)
        elif estimated_height > (available_space * 0.8):
            st.success(f"✅ Cover text length is good (~{estimated_height:.0f}mm, {available_space}mm available)")

    b1, b2, b3 = st.columns([1,1,1])
    cover_style = b1.selectbox("Cover page style", BG_CHOICES, index=1, key="cover_style")
    inner_style = b2.selectbox("Other pages style", BG_CHOICES, index=1, key="inner_style")
    freedom_logo_choice = b3.selectbox("Freedom IT Logo", LOGO_FILES if LOGO_FILES else ["(none found)"], index=0 if LOGO_FILES else 0, key="freedom_logo_choice", help="Select which Freedom IT logo to use on the cover page")
st.markdown("---")
st.subheader("Upload Quotes (2-3 files)")

uploaded = st.file_uploader("Upload Excel files", type=["xlsx", "xls", "xltx"], accept_multiple_files=True, key="excel_files")

if uploaded:
    for idx, f in enumerate(uploaded):
        file_key = f"{f.name}_{f.size}"
        sheets = read_excel_safely(f.read())
        if not sheets:
            continue

        # NEW: Extract snapshot for EACH file (not just first)
        current_snapshot = extract_client_snapshot(sheets)
        
        # NEW: Multi-description extraction
        descs = extract_all_descriptions_from_workbook(sheets)
        desc_md = descs.get("quote_overview", "")
        
        if idx == 0:
            first_snapshot = current_snapshot
            
            # FIXED: Properly assign the extracted descriptions from the workbook
            quote_overview_md = descs.get("quote_overview", "")
            site_desc_md = descs.get("site_services", "")
            phone_desc_md = descs.get("phone_services", "")
            
            try:
                ob_monthly, ob_outright = extract_onboarding_fees(sheets)
                onboarding_monthlyised_final = ob_monthly or 0.0
                onboarding_outright_final = ob_outright or 0.0
                
                # DEBUG: Show what was extracted
                if ob_monthly > 0 or ob_outright > 0:
                    st.success(f"✅ Onboarding fees extracted: Monthlyised=${ob_monthly:.2f}, Outright=${ob_outright:.2f}")
                else:
                    st.warning("⚠️ No onboarding fees found in Overview sheet")
            except Exception as e:
                st.error(f"❌ Error extracting onboarding fees: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                onboarding_monthlyised_final = 0.0
                onboarding_outright_final = 0.0
            # NEW: Extract site services from first file
            site_services_list = extract_site_services(sheets)
            if site_services_list:
                st.info(f"📁  Found {len(site_services_list)} site-based service(s) - will create Infrastructure page")
                # Debug: Show extracted pricing details
                with st.expander("📁 Debug: Site Services Pricing Details", expanded=False):
                    for svc in site_services_list:
                        st.write(f"**{svc['item']}**")
                        st.write(f"  • Unit Price: ${svc['unit_price']:.2f}")
                        st.write(f"  • Quantity: {svc['quantity']}")
                        st.write(f"  • Total: ${svc['total']:.2f}")
                        st.write("---")
            
            # NEW: Extract phone services from first file
            phone_services_list = extract_phone_services(sheets)
            if phone_services_list:
                st.info(f"📞 Found {len(phone_services_list)} phone service(s) - will create Phone Services page")
            
            # NEW: Extract Microsoft 365 licensing from first file
            m365_licenses = extract_m365_licensing(sheets)
            if m365_licenses:
                total_m365_qty = sum(lic['quantity'] for lic in m365_licenses)
                total_m365_cost = sum(lic['total'] for lic in m365_licenses)
                st.info(f"💼 Found {len(m365_licenses)} M365 license type(s) - {total_m365_qty} total licenses")
                
                with st.expander("📋 M365 Licensing Details", expanded=False):
                    for lic in m365_licenses:
                        st.write(f"**{lic['item']}** ({lic['unit_type']})")
                        st.write(f"  • Quantity: {lic['quantity']}")
                        st.write(f"  • Per User: ${lic['unit_price']:.2f}")
                        st.write(f"  • Monthly Total: ${lic['total']:.2f}")
                        st.write("---")
                    st.write(f"**Total M365 Monthly Cost:** ${total_m365_cost:.2f}")


        monthly_support_ex = resolve_monthly_support_ex_gst(sheets)
        monthly_services_ex = resolve_monthly_services_ex_gst(sheets, monthly_support_ex)

        # Save support pricing from first file for PDF generation (as fallback)
        if idx == 0:
            monthly_support_ex_final = monthly_support_ex

        # Extract included services (backwards compatible - returns tuples)
        services = extract_included_services(sheets)
        
        # NEW: Extract included services as items (for services_from_quote compatibility)
        services_items = extract_included_services_as_items(sheets)

        img_bytes = None
        if "comparison" in sheets:
            img_bytes = build_cover_comparison_image()

        quote_name = f.name.replace(".xlsx", "").replace(".xls", "")
        if " - " in quote_name:
            quote_name = quote_name.split(" - ")[-1].strip()

        # FIXED: Store per-quote support amount AND per-quote snapshot data
        # FIXED: Add "items" key so services_from_quote() can extract services
        quotes_data.append({
            "name": quote_name,
            "desc": desc_md,
            "ex_total": monthly_services_ex,
            "support_ex": monthly_support_ex,  # Per-quote support amount
            "proposal_type": current_snapshot.get("proposal_type"),  # Per-quote type
            "hourly_rate": current_snapshot.get("hourly_rate"),  # Per-quote hourly rate
            "rate_display_type": current_snapshot.get("rate_display_type"),  # Per-quote display type
            "img_bytes": img_bytes,
            "services_included": services,  # Backwards compatible format
            "items": services_items  # NEW: Items format for services_from_quote()
        })

        included_raw = to_bullets(desc_md)
        for p in included_raw:
            if p and p not in merged_included_points:
                merged_included_points.append(p)

if quotes_data:
    st.success(f"Loaded {len(quotes_data)} quote(s) successfully!")
    
    if first_snapshot:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Client", first_snapshot.get("client") or "-")
        c2.metric("Sites", first_snapshot.get("sites") or "-")
        c3.metric("Devices", first_snapshot.get("devices") or "-")
        c4.metric("Users", first_snapshot.get("users") or "-")
        
        # Show service accounts if present
        if first_snapshot.get("service_accounts"):
            st.info(f"📁  Service Accounts: {first_snapshot['service_accounts']}")
        
        # Show site services count
        if site_services_list:
            st.info(f"📁  Site Infrastructure Services: {len(site_services_list)} items")
        
        # Show phone users if present
        if first_snapshot.get("phone_users"):
            st.info(f"📞 Phone Users: {first_snapshot['phone_users']}")
        
        # Show phone services count
        if phone_services_list:
            st.info(f"📞 Phone Services: {len(phone_services_list)} items")
        
        # Show onboarding fees if present
        if onboarding_monthlyised_final > 0 or onboarding_outright_final > 0:
            onb_parts = []
            if onboarding_monthlyised_final > 0:
                onb_parts.append(f"Monthlyised: ${onboarding_monthlyised_final:,.2f}")
            if onboarding_outright_final > 0:
                onb_parts.append(f"Outright: ${onboarding_outright_final:,.2f}")
            st.info(f"💼 Company Onboarding Fee - {' | '.join(onb_parts)}")

    for idx, q in enumerate(quotes_data):
        with st.expander(f"Quote {idx+1}: {q['name']}", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Monthly Services (ex GST):** ${q['ex_total']:,.2f}")
            with col2:
                q_support = float(q.get('support_ex', 0.0) or 0.0)
                if q_support > 0:
                    st.markdown(f"**Monthly Support (ex GST):** ${q_support:,.2f}")
                else:
                    st.markdown(f"**Monthly Support (ex GST):** $0.00 *(Not included)*")
            
            if q["desc"].strip():
                st.markdown("**Description:**")
                st.markdown(q["desc"])
            services = q["services_included"]
            if services:
                st.markdown(f"**Included Services ({len(services)}):**")
                for cat, itm in services[:10]:
                    st.caption(f"• {itm}")
                if len(services) > 10:
                    st.caption(f"... and {len(services)-10} more")

st.markdown("---")

with st.expander("PDF Settings", expanded=False):
    gst_rate = st.slider("GST rate (%)", min_value=0, max_value=20, value=10, step=1) / 100.0
    combine_toggle = st.checkbox("Combine services & support pages (when only 1 quote)", value=False)
    
    # Display support status per quote
    st.markdown("---")
    st.markdown("**Support Page Settings:**")
    
    if quotes_data:
        # Show support amount for each quote
        st.markdown("**Per-Quote Support Amounts:**")
        for idx, q in enumerate(quotes_data):
            q_support = float(q.get('support_ex', 0.0) or 0.0)
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{q['name']}:**")
            with col2:
                if q_support > 0:
                    st.success(f"${q_support:,.2f}")
                else:
                    st.info("$0.00")
        
        # Overall support page status
        has_any_support = any(float(q.get('support_ex', 0.0) or 0.0) > 0 for q in quotes_data)
        st.markdown("---")
        if has_any_support:
            st.success("• Support page will be included (at least one quote has support)")
        else:
            st.info("ℹ️ No support page (all quotes have $0 support)")
    else:
        st.info("Upload quotes to see support settings")
    
    # Manual override (now affects all quotes that have $0 support)
    col1, col2 = st.columns([2, 1])
    with col2:
        include_support_override = st.checkbox("Force include support page", value=False, 
                                               help="Check this to include the support page even if all quotes have $0 support")
    
    # Apply override warning
    if include_support_override and quotes_data:
        zero_support_quotes = [q['name'] for q in quotes_data if float(q.get('support_ex', 0.0) or 0.0) == 0]
        if zero_support_quotes:
            st.warning(f"⚠️ Override active: Support page will be included despite these quotes having $0: {', '.join(zero_support_quotes)}")

cover_bg_path = resolve_bg(st.session_state.get("cover_style", "None"))
inner_bg_path = resolve_bg(st.session_state.get("inner_style", "None"))
# FIXED: Safe access to LOGO_FILES with fallback
freedom_logo_path = st.session_state.get("freedom_logo_choice", LOGO_FILES[0] if LOGO_FILES else None)

# Check if footer slogan exists
footer_slogan_path = FOOTER_SLOGAN_FILE if os.path.exists(FOOTER_SLOGAN_FILE) else None

client_logo_bytes = client_logo_file.read() if client_logo_file else None
cover_overview_bytes = overview_img_file.read() if overview_img_file else None

# monthly_support_ex is already calculated in the loop above; keep first file's value




# === V6.0 NEW: Display Nerdio estimate if loaded ===
if st.session_state.nerdio_estimate:
    st.markdown("---")
    st.subheader("📊 Azure Virtual Desktop Cost Estimate")
    
    est = st.session_state.nerdio_estimate
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Cost/User/Month", est.cost_per_user)
    with col2:
        st.metric("Total Monthly", f"${est.cost_per_month:,.2f}")
    with col3:
        st.metric("Users", est.users_count)
    with col4:
        st.metric("Region", est.region)
    
    # Cost breakdown
    with st.expander("💰 Cost Breakdown", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Azure", f"${est.azure_cost:,.2f}")
        with col2:
            st.metric("Nerdio", f"${est.nerdio_cost:,.2f}")
        with col3:
            st.metric("Margin", f"${est.margin_total:,.2f}")
    
    # Desktop pools
    if est.desktop_pools:
        with st.expander("🖥️ Desktop Pools", expanded=False):
            for pool in est.desktop_pools:
                st.markdown(f"**{pool['name']}**")
                cols = st.columns(5)
                cols[0].caption(f"Users: {pool['users']}")
                cols[1].caption(f"VM: {pool['vm_size']}")
                cols[2].caption(f"{pool['cores']}C/{pool['memory']}GB")
                cols[3].caption(f"Hosts: {pool['min_desktops']}-{pool['max_desktops']}")
                cols[4].caption(f"Hours: {pool['min_hours']}-{pool['max_hours']}")
                st.markdown("---")
    
    # Add to PDF option
    include_nerdio = st.checkbox(
        "📄 Include Nerdio estimate in PDF",
        value=True,
        help="Add AVD cost estimate as a page in the generated PDF"
    )

# --- Live PDF Preview ---
# --- V6.0 ENHANCED: PDF Preview with Page Ordering ---
if st.button("Preview PDF", type="secondary", disabled=not uploaded):
    if not quotes_data:
        st.error("No quotes loaded!")
    else:
        with st.spinner("Rendering preview..."):
            try:
                effective_support = monthly_support_ex_final
                if 'include_support_override' in locals() and include_support_override and effective_support == 0:
                    effective_support = 0.01
                pdf_preview = build_pdf(
                    client_name=client_name,
                    cover_md=cover_md,
                    cover_overview_img=cover_overview_bytes,
                    client_logo=client_logo_bytes,
                    freedom_logo_path=freedom_logo_path,
                    quotes=quotes_data,
                    cover_bg_path=cover_bg_path,
                    inner_bg_path=inner_bg_path,
                    snapshot=first_snapshot,
                    monthly_support_ex=effective_support,
                    gst_rate=gst_rate,
                    included_points=merged_included_points,
                    combine_support_and_services=combine_toggle,
                    site_services=site_services_list,
                    phone_services=phone_services_list,
                    footer_slogan_path=footer_slogan_path,
                    site_desc_md=site_desc_md,
                    phone_desc_md=phone_desc_md,
                    onboarding_monthlyised=onboarding_monthlyised_final,
                    onboarding_outright=onboarding_outright_final,
                    doc_cfg=doc_cfg,
                    nerdio_estimate=st.session_state.nerdio_estimate if (st.session_state.nerdio_estimate and 'include_nerdio' in locals() and include_nerdio) else None
                )
                st.success("✅ PDF preview generated!")
                st.session_state.preview_pdf_bytes = pdf_preview
                
                # V6.0 NEW: Page ordering interface
                if st.session_state.preview_mode and PDF_SUPPORT:
                    st.markdown("---")
                    st.subheader("📄 Page Preview & Ordering")
                    
                    main_reader = PdfReader(io.BytesIO(pdf_preview))
                    all_pages = []
                    
                    # Add main PDF pages
                    for page_num in range(len(main_reader.pages)):
                        all_pages.append({
                            "source": "main",
                            "page": page_num,
                            "title": f"Generated Page {page_num + 1}"
                        })
                    
                    # Add uploaded files
                    for file_idx, file_data in enumerate(st.session_state.uploaded_additional_files):
                        if file_data["type"] == "pdf":
                            reader = PdfReader(io.BytesIO(file_data["bytes"]))
                            for page_num in range(len(reader.pages)):
                                all_pages.append({
                                    "source": "upload",
                                    "file_idx": file_idx,
                                    "page": page_num,
                                    "title": f"{file_data['name']} - Page {page_num + 1}",
                                    "file_data": file_data
                                })
                        else:
                            all_pages.append({
                                "source": "upload",
                                "file_idx": file_idx,
                                "page": 0,
                                "title": file_data["name"],
                                "file_data": file_data
                            })
                    
                    st.session_state.all_pages = all_pages
                    st.info(f"Total pages: {len(all_pages)}")
                    
                    if len(all_pages) > 0:
                        st.markdown("**Reorder pages by changing position numbers:**")
                        cols = st.columns(4)
                        new_order = []
                        
                        for idx, page in enumerate(all_pages):
                            col = cols[idx % 4]
                            with col:
                                st.markdown(f"**{page['title']}**")
                                st.caption(f"Source: {page['source']}")
                                position = st.number_input(
                                    "Position",
                                    min_value=1,
                                    max_value=len(all_pages),
                                    value=idx + 1,
                                    key=f"page_pos_{idx}"
                                )
                                new_order.append((position, idx, page))
                        
                        new_order.sort()
                        st.session_state.page_order = [page for _, _, page in new_order]
                        
                        with st.expander("📋 Final page order", expanded=True):
                            for pos, (_, _, page) in enumerate(new_order):
                                st.write(f"{pos + 1}. {page['title']}")
                
                # Download preview
                st.download_button(
                    label="🔥 Download Preview (PDF)",
                    data=pdf_preview,
                    file_name="preview.pdf",
                    mime="application/pdf",
                    type="primary"
                )
            except Exception as e:
                st.error(f"Preview failed: {e}")
                import traceback
                st.code(traceback.format_exc())

# --- V6.0 ENHANCED: Generate PDF with File Merging ---
if st.button("Generate PDF", type="primary", disabled=not uploaded):
    if not quotes_data:
        st.error("No quotes loaded!")
    else:
        with st.spinner("Building PDF..."):
            try:
                # Apply support override if needed
                effective_support = monthly_support_ex_final
                if include_support_override and effective_support == 0:
                    effective_support = 0.01
                
                pdf_bytes = build_pdf(
                    client_name=client_name,
                    cover_md=cover_md,
                    cover_overview_img=cover_overview_bytes,
                    client_logo=client_logo_bytes,
                    freedom_logo_path=freedom_logo_path,
                    quotes=quotes_data,
                    cover_bg_path=cover_bg_path,
                    inner_bg_path=inner_bg_path,
                    snapshot=first_snapshot,
                    monthly_support_ex=effective_support,
                    gst_rate=gst_rate,
                    included_points=merged_included_points,
                    combine_support_and_services=combine_toggle,
                    site_services=site_services_list,
                    phone_services=phone_services_list,
                    footer_slogan_path=footer_slogan_path,
                    site_desc_md=site_desc_md,
                    phone_desc_md=phone_desc_md,
                    onboarding_monthlyised=onboarding_monthlyised_final,
                    onboarding_outright=onboarding_outright_final,
                    doc_cfg=doc_cfg,
                    nerdio_estimate=st.session_state.nerdio_estimate if (st.session_state.nerdio_estimate and 'include_nerdio' in locals() and include_nerdio) else None
                )
                
                # V6.0 NEW: Merge with uploaded files if any
                if st.session_state.uploaded_additional_files and PDF_SUPPORT:
                    if st.session_state.page_order:
                        # Use custom order
                        pdf_bytes = merge_pdfs_with_order(pdf_bytes, st.session_state.page_order)
                        st.info(f"✅ Merged {len(st.session_state.page_order)} pages in custom order")
                    else:
                        # Default: append files to end
                        writer = PdfWriter()
                        main_reader = PdfReader(io.BytesIO(pdf_bytes))
                        for page in main_reader.pages:
                            writer.add_page(page)
                        
                        for file_data in st.session_state.uploaded_additional_files:
                            if file_data["type"] == "pdf":
                                reader = PdfReader(io.BytesIO(file_data["bytes"]))
                                for page in reader.pages:
                                    writer.add_page(page)
                        
                        output = io.BytesIO()
                        writer.write(output)
                        pdf_bytes = output.getvalue()
                        st.info(f"✅ Appended {len(st.session_state.uploaded_additional_files)} file(s)")
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"{client_name.replace(' ', '_')}_Proposal_{timestamp}.pdf" if client_name else f"Proposal_{timestamp}.pdf"
                
                st.success("PDF generated successfully!")
                st.download_button(
                    label="Download PDF",
                    data=pdf_bytes,
                    file_name=fname,
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(f"Failed to generate PDF: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                import traceback
                st.code(traceback.format_exc())
# ---------------------------
# Template loader for markdown sections
# ---------------------------
from pathlib import Path as _Path

def load_template_markdown(slug: str, templates_dir: str | _Path | None):
    """Load markdown file /templates/<slug>.md. Returns (title, markdown).
    First line beginning with '# ' becomes the title. If missing, slug in Title Case is used."""
    base = _Path(templates_dir) if templates_dir else _Path("./templates")
    pth = base / f"{slug}.md"
    if not pth.exists():
        title = slug.replace("_"," ").title()
        md = f"# {title}\n\n_Missing template: {pth}_"
        return title, md
    raw = pth.read_text(encoding="utf-8").strip()
    lines = raw.splitlines()
    title = lines[0][2:].strip() if lines and lines[0].startswith("# ") else slug.replace("_"," ").title()
    return title, raw