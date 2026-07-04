import os
import logging
import asyncio
import requests
from io import BytesIO
from datetime import datetime
from typing import Optional, Tuple  # ✅ FIXED: Added Optional import

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)
from telegram.constants import ParseMode

# ============================================
# CONFIGURATION & SETUP
# ============================================

# Environment variables (set in Railway)
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
STABILITY_API_KEY = os.environ.get("STABILITY_API_KEY")

# Optional configuration
ALLOWED_USERS = os.environ.get("ALLOWED_USERS", "*")
ADMIN_USER_ID = os.environ.get("ADMIN_USER_ID", "")
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "")

# API Configuration
API_HOST = "https://api.stability.ai"
API_VERSION = "v1"
GENERATION_ENDPOINT = f"{API_HOST}/{API_VERSION}/generation/stable-diffusion-xl-1024-v1-0/text-to-image"

# Bot settings
MAX_PROMPT_LENGTH = 1000
DEFAULT_STEPS = 30
DEFAULT_CFG_SCALE = 7

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ============================================
# HELPER FUNCTIONS
# ============================================

def is_user_allowed(user_id: int) -> bool:
    """Check if user is authorized to use the bot."""
    if ALLOWED_USERS == "*":
        return True
    try:
        allowed_list = [int(uid.strip()) for uid in ALLOWED_USERS.split(",") if uid.strip()]
        return user_id in allowed_list
    except ValueError:
        logger.error(f"Invalid ALLOWED_USERS format: {ALLOWED_USERS}")
        return False

def validate_prompt(prompt: str) -> Tuple[bool, str]:
    """Validate and sanitize the user prompt."""
    if not prompt or not prompt.strip():
        return False, "Please provide a description for the image."
    
    if len(prompt) > MAX_PROMPT_LENGTH:
        return False, f"Prompt is too long ({len(prompt)} chars). Maximum {MAX_PROMPT_LENGTH} characters."
    
    clean_prompt = " ".join(prompt.strip().split())
    return True, clean_prompt

def generate_image(prompt: str, steps: int = DEFAULT_STEPS, cfg_scale: float = DEFAULT_CFG_SCALE) -> Optional[bytes]:
    """
    Generate an image using Stability AI API.
    Returns image data as bytes or None on failure.
    """
    if not STABILITY_API_KEY:
        logger.error("STABILITY_API_KEY is not set!")
        return None
    
    headers = {
        "Authorization": f"Bearer {STABILITY_API_KEY}",
        "Accept": "image/png",
        "Content-Type": "application/json",
    }
    
    payload = {
        "text_prompts": [{"text": prompt}],
        "cfg_scale": cfg_scale,
        "height": 1024,
        "width": 1024,
        "samples": 1,
        "steps": steps,
        "style_preset": "photographic",
    }

    try:
        logger.info(f"Generating image for prompt: {prompt[:50]}...")
        response = requests.post(
            GENERATION_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=60,
        )
        
        if response.status_code == 200:
            logger.info("✅ Image generated successfully")
            return response.content
        else:
            error_msg = f"API Error {response.status_code}: {response.text}"
            logger.error(error_msg)
            return None
            
    except requests.exceptions.Timeout:
        logger.error("API request timed out")
        return None
    except requests.exceptions.ConnectionError:
        logger.error("Connection error to Stability AI API")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in generate_image: {str(e)}")
        return None

async def send_typing_action(update: Update) -> None:
    """Show typing indicator while processing."""
    await update.effective_message.chat.send_action(action="typing")

def get_help_text() -> str:
    """Return the help message."""
    return (
        "🎨 *AI Image Generation Bot*\n\n"
        "Generate stunning images from text descriptions using AI.\n\n"
        "*How to use:*\n"
        "1. Send me a descriptive text of what you want to see\n"
        "2. Wait a few seconds for the AI to create your image\n"
        "3. Receive the generated image directly in chat\n\n"
        "*Tips for better results:*\n"
        "• Be specific about style (photorealistic, anime, oil painting)\n"
        "• Include lighting, colors, and composition details\n"
        "• Mention atmosphere (moody, bright, dark)\n\n"
        "*Commands:*\n"
        "/start - Welcome message\n"
        "/help - Show this help\n"
        "/info - Bot information\n\n"
        "*Example prompts:*\n"
        "• 'A photorealistic cat wearing a crown, dramatic lighting'\n"
        "• 'Cyberpunk city at night, neon lights, digital art'\n"
        "• 'Oil painting of a mountain lake at sunset'\n\n"
        "⭐ *Try:* 'style: photographic' or 'style: anime' in your prompt"
    )

def get_start_text(first_name: str) -> str:
    """Return the welcome message."""
    return (
        f"👋 Welcome, {first_name}!\n\n"
        f"I'm *@ImageGenProbot*, your AI image generation assistant.\n"
        f"I use cutting-edge AI to turn your imagination into images.\n\n"
        f"✨ Just send me any text description and I'll create a unique image for you!\n\n"
        f"Use /help to learn how to get the best results.\n\n"
        f"Ready to create something amazing? 🚀"
    )

# ============================================
# BOT COMMAND HANDLERS
# ============================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    user = update.effective_user
    
    if not is_user_allowed(user.id):
        await update.message.reply_text("❌ You are not authorized to use this bot.")
        return
    
    welcome_text = get_start_text(user.first_name)
    
    keyboard = [
        [
            InlineKeyboardButton("📖 Help", callback_data="help"),
            InlineKeyboardButton("ℹ️ Info", callback_data="info"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        welcome_text,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=reply_markup,
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    user = update.effective_user
    
    if not is_user_allowed(user.id):
        await update.message.reply_text("❌ You are not authorized to use this bot.")
        return
    
    await update.message.reply_text(
        get_help_text(),
        parse_mode=ParseMode.MARKDOWN,
    )

async def info_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /info command."""
    user = update.effective_user
    
    if not is_user_allowed(user.id):
        await update.message.reply_text("❌ You are not authorized to use this bot.")
        return
    
    info_text = (
        f"🤖 *Bot Information*\n\n"
        f"*Name:* @ImageGenProbot\n"
        f"*Version:* 2.0.0\n"
        f"*Engine:* Stable Diffusion XL (SDXL)\n"
        f"*Resolution:* 1024×1024\n"
        f"*Status:* 🟢 Online\n\n"
        f"*Features:*\n"
        f"✅ Text-to-image generation\n"
        f"✅ Multiple art styles\n"
        f"✅ High-quality 1024px images\n"
        f"✅ Fast response time\n\n"
        f"Made with ❤️ using Python and Telegram Bot API"
    )
    
    await update.message.reply_text(info_text, parse_mode=ParseMode.MARKDOWN)

# ============================================
# MESSAGE HANDLERS
# ============================================

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle text messages for image generation."""
    user = update.effective_user
    
    if not is_user_allowed(user.id):
        await update.message.reply_text("❌ You are not authorized to use this bot.")
        return
    
    prompt = update.message.text
    
    # Validate prompt
    is_valid, message = validate_prompt(prompt)
    if not is_valid:
        await update.message.reply_text(f"❌ {message}")
        return
    
    # Show typing indicator
    await send_typing_action(update)
    
    # Send initial processing message
    processing_msg = await update.message.reply_text(
        f"🎨 Generating image...\n\n"
        f"*Prompt:* {prompt[:100]}{'...' if len(prompt) > 100 else ''}\n\n"
        f"⏳ Please wait, this may take 5-15 seconds...",
        parse_mode=ParseMode.MARKDOWN,
    )
    
    try:
        # Generate the image
        image_data = await asyncio.to_thread(generate_image, prompt)
        
        if image_data:
            # Delete processing message
            await processing_msg.delete()
            
            # Send the generated image
            await update.message.reply_photo(
                photo=BytesIO(image_data),
                caption=(
                    f"✅ *Image Generated!*\n\n"
                    f"*Prompt:* {prompt[:200]}{'...' if len(prompt) > 200 else ''}\n"
                    f"*Size:* 1024×1024\n"
                    f"*Engine:* SDXL\n\n"
                    f"🔄 To generate another, just send a new prompt!"
                ),
                parse_mode=ParseMode.MARKDOWN,
            )
        else:
            # Failed to generate
            await processing_msg.edit_text(
                "❌ *Generation Failed*\n\n"
                "I couldn't generate an image. This might be due to:\n"
                "• API service unavailable\n"
                "• API key issue\n"
                "• Prompt content violation\n\n"
                "Please try again in a few moments or with a different prompt.",
                parse_mode=ParseMode.MARKDOWN,
            )
            
    except Exception as e:
        logger.error(f"Error in handle_text: {str(e)}")
        await processing_msg.edit_text(
            "❌ *Unexpected Error*\n\n"
            "Something went wrong during image generation.\n"
            "Please try again later.",
            parse_mode=ParseMode.MARKDOWN,
        )

# ============================================
# CALLBACK QUERY HANDLERS
# ============================================

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline keyboard button presses."""
    query = update.callback_query
    await query.answer()
    
    user = update.effective_user
    
    if not is_user_allowed(user.id):
        await query.edit_message_text("❌ You are not authorized to use this bot.")
        return
    
    callback_data = query.data
    
    if callback_data == "help":
        await query.edit_message_text(
            get_help_text(),
            parse_mode=ParseMode.MARKDOWN,
        )
    elif callback_data == "info":
        info_text = (
            f"🤖 *Bot Information*\n\n"
            f"*Name:* @ImageGenProbot\n"
            f"*Version:* 2.0.0\n"
            f"*Engine:* Stable Diffusion XL\n"
            f"*Status:* 🟢 Online\n\n"
            f"Made with ❤️"
        )
        await query.edit_message_text(info_text, parse_mode=ParseMode.MARKDOWN)
    else:
        await query.edit_message_text(f"Unknown option: {callback_data}")

# ============================================
# ERROR HANDLERS
# ============================================

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle errors in the bot."""
    logger.error(f"Update {update} caused error {context.error}")
    
    try:
        if update and update.effective_message:
            await update.effective_message.reply_text(
                "⚠️ An unexpected error occurred. Please try again later."
            )
    except Exception as e:
        logger.error(f"Failed to send error message: {e}")

# ============================================
# MAIN APPLICATION
# ============================================

def main() -> None:
    """Start the bot."""
    # Validate environment variables
    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN environment variable not set!")
        return
    
    if not STABILITY_API_KEY:
        logger.warning("⚠️ STABILITY_API_KEY not set! Image generation will fail.")
    else:
        logger.info("✅ STABILITY_API_KEY found")
    
    # Create application
    application = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .build()
    )
    
    # Register command handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("info", info_command))
    
    # Register message handlers
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    
    # Register callback query handler
    application.add_handler(CallbackQueryHandler(button_callback))
    
    # Register error handler
    application.add_error_handler(error_handler)
    
    # Start the bot
    logger.info("🚀 Bot is starting...")
    
    # Use webhook if URL is provided (Railway) or polling
    if WEBHOOK_URL:
        logger.info(f"Using webhook: {WEBHOOK_URL}")
        application.run_webhook(
            listen="0.0.0.0",
            port=int(os.environ.get("PORT", 8443)),
            url_path=TELEGRAM_TOKEN,
            webhook_url=f"{WEBHOOK_URL}/{TELEGRAM_TOKEN}",
        )
    else:
        logger.info("Using polling mode...")
        application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
