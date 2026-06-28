"""Generate a small, generic synthetic demo corpus (fictional product: Acme Cloud).

Produces a few PDFs under demo_docs/ so the public repo ships reproducible,
non-confidential demo content for screenshots and trying the pipeline.

    python scripts/make_demo_corpus.py
"""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "demo_docs"
OUT.mkdir(parents=True, exist_ok=True)


def _font(size: int):
    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def make_share_dialog_image() -> Path:
    """Render a simple mock 'Share' dialog so a demo doc has an embedded image
    (exercises the image extraction + vision-captioning pipeline)."""
    img = Image.new("RGB", (660, 380), "#eef1f7")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 660, 52], fill="#34507e")
    d.text((18, 17), "Acme Cloud  -  Share \"Q4-plan.pdf\"", fill="white", font=_font(18))
    d.text((24, 80), "Invite people by email", fill="#222", font=_font(15))
    d.rectangle([24, 110, 470, 146], outline="#9aa6c2", fill="white")
    d.text((34, 119), "alex@example.com", fill="#444", font=_font(14))
    d.rectangle([486, 110, 636, 146], outline="#9aa6c2", fill="white")
    d.text((496, 119), "Can edit  v", fill="#444", font=_font(14))
    d.text((24, 178), "Or copy a link", fill="#222", font=_font(15))
    d.rectangle([24, 208, 636, 244], outline="#9aa6c2", fill="white")
    d.text((34, 217), "https://acme.cloud/s/9fA2qZ   (view only)", fill="#3357a8", font=_font(14))
    d.rectangle([512, 312, 636, 352], fill="#3b6cf6")
    d.text((545, 323), "Share", fill="white", font=_font(15))
    p = OUT / "_share_dialog.png"
    img.save(p)
    return p


class Doc(FPDF):
    def _block(self, text: str, h: float) -> None:
        self.set_x(self.l_margin)
        self.multi_cell(self.epw, h, text, new_x="LMARGIN", new_y="NEXT")

    def doc_title(self, text: str) -> None:
        self.set_font("Helvetica", "B", 20)
        self._block(text, 10)
        self.ln(2)

    def h2(self, text: str) -> None:
        self.ln(2)
        self.set_font("Helvetica", "B", 14)
        self._block(text, 8)
        self.ln(1)

    def para(self, text: str) -> None:
        self.set_font("Helvetica", "", 11)
        self._block(text, 6)
        self.ln(1)

    def steps(self, items: list[str]) -> None:
        self.set_font("Helvetica", "", 11)
        for i, it in enumerate(items, 1):
            self._block(f"{i}. {it}", 6)
        self.ln(1)

    def img(self, path, w: float = 150) -> None:
        self.ln(2)
        self.set_x(self.l_margin)
        self.image(str(path), w=w)
        self.ln(2)


def build(filename: str, title: str, blocks: list) -> None:
    pdf = Doc()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.doc_title(title)
    for kind, payload in blocks:
        if kind == "h2":
            pdf.h2(payload)
        elif kind == "p":
            pdf.para(payload)
        elif kind == "steps":
            pdf.steps(payload)
        elif kind == "img":
            pdf.img(payload)
        elif kind == "pagebreak":
            pdf.add_page()
    pdf.output(str(OUT / filename))
    print("wrote", OUT / filename)


build(
    "acme_cloud_getting_started.pdf",
    "Acme Cloud - Getting Started Guide",
    [
        ("p", "Acme Cloud is a cloud storage and team collaboration platform. This guide covers creating your account, signing in, and sharing your first files."),
        ("h2", "Creating an account"),
        ("steps", [
            "Go to the sign-up page and enter your work email address.",
            "Choose a strong password, or continue with single sign-on (SSO) if your organization has it enabled.",
            "Verify your email by clicking the link we send you.",
            "Create your first workspace and invite teammates by email.",
        ]),
        ("h2", "Signing in with single sign-on"),
        ("p", "If your administrator has configured SSO, click 'Sign in with SSO' and enter your company domain. You will be redirected to your identity provider to authenticate. Two-factor authentication (2FA) may also be required."),
        ("h2", "Uploading and organizing files"),
        ("steps", [
            "Open a workspace and click Upload, or drag files into the browser window.",
            "Create folders to organize files; you can nest folders to any depth.",
            "Use a shared drive when a whole team needs access to the same files.",
            "Right-click any file to rename, move, download, or view its version history.",
        ]),
        ("h2", "Sharing a file"),
        ("p", "To share a file, select it and click Share. You can invite specific people by email, or create a link. Link permissions can be set to view-only or edit, and you can require sign-in or set an expiry date."),
    ],
)

build(
    "acme_cloud_admin_security.pdf",
    "Acme Cloud - Administrator & Security Guide",
    [
        ("p", "This guide is for workspace administrators. It explains user roles, authentication, access control, and audit logging."),
        ("h2", "User roles"),
        ("p", "Acme Cloud has three roles: Member (can read and edit files they have access to), Manager (can manage a workspace and its members), and Admin (can manage the whole organization, billing, and security settings)."),
        ("h2", "Enforcing two-factor authentication"),
        ("steps", [
            "Open Admin Settings and select Security.",
            "Turn on 'Require two-factor authentication (2FA)' for all members.",
            "Members will be prompted to enrol an authenticator app on next sign-in.",
            "You can reset a member's 2FA from the Members page if they lose their device.",
        ]),
        ("h2", "Configuring single sign-on"),
        ("p", "Acme Cloud supports SAML and OIDC single sign-on. In Admin Settings > SSO, enter your identity provider's metadata URL, map the email and name attributes, and assign a default role for newly provisioned users."),
        ("h2", "Access control lists"),
        ("p", "Every file and folder has an access control list (ACL) that grants permissions to users or groups. Permissions inherit from the parent folder unless you override them. Use groups rather than individual users to keep ACLs maintainable."),
        ("h2", "API keys and webhooks"),
        ("p", "Generate an API key under Settings > Developer to call the Acme Cloud API. Keep API keys secret and rotate them periodically. Configure a webhook to receive a callback when files are created, updated, or deleted."),
        ("h2", "Audit log and data retention"),
        ("p", "The audit log records sign-ins, permission changes, and file activity. Set a data retention policy to automatically delete files in the trash after a fixed number of days, or to retain audit records for compliance."),
    ],
)

build(
    "acme_cloud_billing.pdf",
    "Acme Cloud - Billing and Plans",
    [
        ("p", "This document explains the available plans, how to upgrade, and how billing works."),
        ("h2", "Plans"),
        ("p", "Acme Cloud offers three plans. Free includes 5 GB of storage for a single user. Team adds shared drives, 1 TB per user, and admin controls. Business adds SSO, advanced audit logs, and unlimited version history."),
        ("h2", "Upgrading your plan"),
        ("steps", [
            "Open Admin Settings and select Billing.",
            "Choose the Team or Business plan and the number of seats.",
            "Enter a payment method (credit card or invoice for annual plans).",
            "Confirm; the new plan takes effect immediately and is prorated.",
        ]),
        ("h2", "Invoices and payment methods"),
        ("p", "Invoices are issued monthly or annually and are available under Billing > Invoices as PDF downloads. You can add multiple payment methods and set one as the default. Annual plans can be paid by bank transfer."),
        ("h2", "Cancelling or downgrading"),
        ("p", "You can downgrade or cancel at any time from Billing. Your plan remains active until the end of the current billing period, after which storage above the new limit becomes read-only until you remove files."),
    ],
)

build(
    "acme_cloud_troubleshooting.pdf",
    "Acme Cloud - Troubleshooting and FAQ",
    [
        ("p", "Common problems and frequently asked questions. If your issue is not listed, contact support from the Help menu."),
        ("h2", "I cannot sign in"),
        ("p", "First confirm your email and password are correct. If your organization uses SSO, use the 'Sign in with SSO' button instead of a password. If two-factor authentication fails, make sure your device clock is accurate, or ask an admin to reset your 2FA."),
        ("h2", "My upload failed"),
        ("steps", [
            "Check that the file is under your plan's per-file size limit.",
            "Confirm you have enough remaining storage in the workspace.",
            "Disable browser extensions that may block uploads, then retry.",
            "For large uploads, use the desktop sync app instead of the browser.",
        ]),
        ("h2", "Files are not syncing"),
        ("p", "Make sure the desktop sync app is running and signed in. Check that the workspace is selected for sync and that you have permission to the folder. Pausing and resuming sync often resolves a stuck file."),
        ("h2", "Frequently asked questions"),
        ("p", "Q: Can I recover a deleted file? A: Yes, deleted files stay in the trash and can be restored until the retention policy removes them."),
        ("p", "Q: How do I get an API key? A: An admin can create one under Settings > Developer; see the Administrator guide."),
        ("p", "Q: Does Acme Cloud support single sign-on? A: Yes, SAML and OIDC SSO are available on the Business plan."),
    ],
)

# A doc with an embedded image — exercises image extraction + vision captioning,
# so the corpus has chunks that contain a figure (visible on the Images page).
_share_img = make_share_dialog_image()
build(
    "acme_cloud_visual_guide.pdf",
    "Acme Cloud - Visual Guide: Sharing",
    [
        ("p", "This visual guide shows the Share dialog used to share a file with people "
              "or via a link. Select a file and click Share to open it."),
        ("p", "The Share dialog lets you invite people by email with a permission level "
              "(view-only or edit), or copy a link. The screenshot below shows the dialog:"),
        ("pagebreak", None),
        ("p", "Figure 1 - The Acme Cloud Share dialog:"),
        ("img", _share_img),
        ("p", "Set the email permission to 'Can edit' to let collaborators change the file, "
              "or copy the view-only link to share read access. Links can require sign-in or expire."),
    ],
)
try:
    _share_img.unlink()
except Exception:
    pass

print("done")
