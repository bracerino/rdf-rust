"""Sidebar branding for the online app: supporter logos and the journal articles.

Mirrors more_funct/branding.py of XRDlicious, so both apps look the same.
"""

import base64
from pathlib import Path

_IMAGES = Path(__file__).resolve().parent / "images"

# (name, website, logo file, logo height in px)
SUPPORTERS = [
    ("FZU – Institute of Physics of the Czech Academy of Sciences",
     "https://www.fzu.cz/en", "fzu_logo.svg", 34),
    ("IMPLANT group", "https://implant.fs.cvut.cz/", "implant_logo.png", 44),
]

# Both articles appeared in the Journal of Applied Crystallography. The badge
# links to the XRDlicious article, as in XRDlicious itself; RDF-Rust is part of
# the XRD-Rust project, so that article is listed under it.
ARTICLES = [
    ("XRDlicious (2025)", "https://doi.org/10.1107/S1600576725005370",
     "Lebeda et al. (2025). XRDlicious: an interactive web-based platform for online "
     "calculation of diffraction patterns and radial distribution functions from "
     "crystal structures. J. Appl. Cryst. 58"),
    ("XRD-Rust / RDF-Rust (2026)",
     "https://journals.iucr.org/j/issues/2026/04/00/hat5023/index.html",
     "Lebeda et al. (2026). Rust-accelerated powder X-ray diffraction simulation for "
     "high-throughput and machine-learning-driven materials science. J. Appl. Cryst. 59"),
]
IUCR_MARK_FILES = ("iucr_journals_logo_spaces.svg", "iucr_journals_logo_spaces.png",
                   "iucr_journals_logo_spaces.webp")


# GitHub mark (Octicons "mark-github", MIT license). Inherits the text color and
# size of the surrounding line, so it fits into any sentence.
GITHUB_ICON = (
    '<svg viewBox="0 0 16 16" width="1em" height="1em" fill="currentColor" '
    'aria-hidden="true" style="vertical-align:-0.125em;margin-right:0.25em;">'
    '<path d="M8 0c4.42 0 8 3.58 8 8a8.013 8.013 0 0 1-5.45 7.59c-.4.08-.55-.17'
    '-.55-.38 0-.27.01-1.13.01-2.2 0-.75-.25-1.23-.54-1.48 1.78-.2 3.65-.88 3.65'
    '-3.95 0-.88-.31-1.59-.82-2.15.08-.2.36-1.02-.08-2.12 0 0-.67-.22-2.2.82-.64'
    '-.18-1.32-.27-2-.27-.68 0-1.36.09-2 .27-1.53-1.03-2.2-.82-2.2-.82-.44 1.1'
    '-.16 1.92-.08 2.12-.51.56-.82 1.28-.82 2.15 0 3.06 1.86 3.75 3.64 3.95-.23.2'
    '-.44.55-.51 1.07-.46.21-1.61.55-2.33-.66-.15-.24-.6-.83-1.23-.82-.67.01-.27'
    '.38.01.53.34.19.73.9.82 1.13.16.45.68 1.31 2.69.94 0 .67.01 1.3.01 1.49 0 .21'
    '-.15.45-.55.38A7.995 7.995 0 0 1 0 8c0-4.42 3.58-8 8-8Z"/></svg>'
)


def github_link_html(text="GitHub page",
                     url="https://github.com/bracerino/rdf-rust"):
    return (f'<a href="{url}" target="_blank" style="white-space:nowrap;">'
            f'{GITHUB_ICON}<b>{text}</b></a>')


def render_about(container):
    """The 'Developed by' note, in the look of st.info.

    Written in HTML because st.info cannot hold the inline GitHub icon.
    """
    container.markdown(
        '<div style="background-color:rgba(28,131,225,0.1);color:rgb(0,66,128);'
        'border-radius:8px;padding:16px;line-height:1.6;margin-bottom:1rem;">'
        '🌀 Developed by <a href="https://implant.fs.cvut.cz/" target="_blank">'
        '<b>IMPLANT team</b></a>. Spot a bug or have a feature idea? Let us know at: '
        '<b>lebedmi2@cvut.cz</b>. To compile the full app locally, visit our '
        f'{github_link_html()}. If you use this tool, please cite the 📖 '
        f'<a href="{ARTICLES[0][1]}" target="_blank"><b>article in IUCr</b></a>. '
        '🫶 <a href="https://buymeacoffee.com/bracerino" target="_blank"><b>Donations</b></a> '
        'always appreciated!</div>',
        unsafe_allow_html=True,
    )


def _data_uri(path):
    mime = {".svg": "image/svg+xml", ".webp": "image/webp",
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(path.suffix.lower(), "image/png")
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def _journal_html():
    mark = next((_IMAGES / f for f in IUCR_MARK_FILES if (_IMAGES / f).exists()), None)
    mark_html = (f'<img src="{_data_uri(mark)}" alt="IUCr" '
                 'style="height:30px;width:30px;flex:none;">' if mark else "")
    # without the mark, a coloured bar on the left takes its place
    bar = "" if mark else "border-left:4px solid #1f4e8c;padding-left:8px;"
    badge = (
        '<span style="display:inline-flex;align-items:center;gap:8px;">'
        + mark_html +
        f'<span style="display:inline-block;{bar}padding-top:2px;padding-bottom:2px;'
        'line-height:1.05;font-family:Georgia,\'Times New Roman\',serif;color:#1f4e8c;">'
        '<span style="font-size:0.78em;font-style:italic;">Journal of</span><br>'
        '<span style="font-size:0.95em;font-weight:700;white-space:nowrap;">'
        'Applied Crystallography</span></span></span>')
    _label, url, title = ARTICLES[0]
    links = "".join(
        f'<li><a href="{u}" target="_blank" title="{t}">{label}</a></li>'
        for label, u, t in ARTICLES)
    return (
        '<div style="font-size:0.85em;opacity:0.8;margin:14px 0 6px 0;">Published in</div>'
        f'<a href="{url}" target="_blank" title="{title}" style="text-decoration:none;">'
        + badge + '</a>'
        '<ul style="font-size:0.85em;margin:6px 0 0 0;padding-left:1.2em;">'
        + links + '</ul>')


def render_supporters(container):
    logos = []
    for name, url, fname, height in SUPPORTERS:
        path = _IMAGES / fname
        if not path.exists():
            continue
        logos.append(
            f'<a href="{url}" target="_blank" title="{name}">'
            f'<img src="{_data_uri(path)}" alt="{name}" '
            f'style="height:{height}px;width:auto;"></a>')
    # Keyed so Streamlit tracks the block by key, not by position: otherwise a
    # rerun that adds sidebar content above it (a newly uploaded file) briefly
    # shows the old and the new copy at once.
    box = container.container(key="sidebar_supporters")
    box.markdown("---")
    box.markdown(
        '<div style="font-size:0.85em;opacity:0.8;margin-bottom:6px;">'
        'Supported by <b>FZU</b> and the <b>IMPLANT group</b></div>'
        '<div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;">'
        + "".join(logos) + '</div>' + _journal_html(),
        unsafe_allow_html=True,
    )
