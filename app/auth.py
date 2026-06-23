"""
app/auth.py — Telegram login helpers (QR code and phone number).
"""
from __future__ import annotations

import asyncio
import stat
from getpass import getpass
from pathlib import Path

from telethon import TelegramClient
from telethon.errors import (
    FloodWaitError,
    PhoneNumberBannedError,
    PhoneNumberInvalidError,
    SessionPasswordNeededError,
)


async def _login_qr(client: TelegramClient) -> None:
    """Authenticate via QR code. Auto-refreshes every 20 s. Handles 2FA."""
    try:
        import qrcode
    except ImportError:
        raise RuntimeError("qrcode package not installed. Run: uv add qrcode")

    print("\nOpen Telegram on your phone:")
    print("  Settings → Devices → Link Desktop Device")
    print("Scan the QR code below (auto-refreshes every 20 s).\n")

    while True:
        qr_login = await client.qr_login()
        qr = qrcode.QRCode()
        qr.add_data(qr_login.url)
        qr.make()
        qr.print_ascii(invert=True)

        try:
            await qr_login.wait(timeout=20)
            return
        except SessionPasswordNeededError:
            pwd = getpass("2FA password: ")
            await client.sign_in(password=pwd)
            return
        except Exception:
            print("Refreshing QR…\n")
            continue


async def _login_phone(client: TelegramClient) -> None:
    """Authenticate via phone number + code. Handles SentCodeTypeApp, 2FA, flood."""
    phone = input("Phone number (international format, e.g. +79001234567): ").strip()

    print("\nRequesting code…")
    try:
        result = await client.send_code_request(phone)
    except FloodWaitError as e:
        print(f"\nRate-limited by Telegram. Wait {e.seconds} s ({e.seconds // 60} min) and try again.")
        return
    except PhoneNumberBannedError:
        print("\nThis phone number is banned by Telegram.")
        return
    except PhoneNumberInvalidError:
        print("\nInvalid phone number. Use international format: +12345678901")
        return

    code_type = str(result.type)
    if "SentCodeTypeApp" in code_type:
        print(
            "\nCode type: App (not SMS).\n"
            "Open Telegram on your phone or any device where you are already logged in.\n"
            "Look for a message from the official 'Telegram' account (user 777000).\n"
            "\nIf nothing arrives after 30 s, press Enter to request SMS resend.\n"
            "Alternatively, use QR login:  python -m app login --method qr\n"
        )
        code = input("Enter the 5-digit code (or press Enter to resend as SMS): ").strip()
        if not code:
            try:
                result2 = await client.resend_code(phone, result.phone_code_hash)
                print(f"Resent via {result2.type}. Check your phone.")
                code = input("Enter the code: ").strip()
            except Exception as e:
                print(f"Resend failed: {e}")
                return
    else:
        print(f"\nCode sent via {code_type}. Check your phone.")
        code = input("Enter the code: ").strip()

    if not code:
        print("No code entered — login cancelled.")
        return

    try:
        await client.sign_in(phone, code)
    except SessionPasswordNeededError:
        pwd = getpass("2FA password: ")
        await client.sign_in(password=pwd)
    except Exception as e:
        print(f"Sign-in failed: {type(e).__name__}: {e}")


def _secure_session(session_path: Path) -> None:
    """Set 600 permissions on the .session file (POSIX only, best-effort)."""
    candidate = session_path.parent / (session_path.name + ".session")
    if not candidate.exists():
        candidate = session_path  # some platforms omit the extension
    try:
        if candidate.exists():
            candidate.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except Exception:
        pass


async def ensure_logged_in(
    session_path: Path,
    api_id: int,
    api_hash: str,
    method: str | None = None,
) -> None:
    """
    Connect a TelegramClient and authenticate if needed.

    Args:
        session_path: path passed to TelegramClient (without .session extension).
        api_id: Telegram API ID from my.telegram.org.
        api_hash: Telegram API hash from my.telegram.org.
        method: 'qr' | 'phone' | None (prompt user to choose).
    """
    session_path.parent.mkdir(parents=True, exist_ok=True)
    client = TelegramClient(str(session_path), api_id, api_hash)

    await client.connect()

    if await client.is_user_authorized():
        me = await client.get_me()
        name = me.first_name or ""
        handle = f"@{me.username}" if me.username else f"id={me.id}"
        print(f"Already logged in as {name} ({handle})")
        await client.disconnect()
        _secure_session(session_path)
        return

    if method is None:
        print("\nChoose login method:")
        print("  1) QR code   (recommended — works without SMS)")
        print("  2) Phone number + verification code")
        choice = input("Enter 1 or 2: ").strip()
        method = "qr" if choice == "1" else "phone"

    if method == "qr":
        await _login_qr(client)
    else:
        await _login_phone(client)

    if await client.is_user_authorized():
        me = await client.get_me()
        name = me.first_name or ""
        handle = f"@{me.username}" if me.username else f"id={me.id}"
        print(f"\nSuccess! Logged in as {name} ({handle})")
        _secure_session(session_path)
    else:
        print("\nLogin did not complete — not authorized.")

    await client.disconnect()
