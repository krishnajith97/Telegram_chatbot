import logging
import asyncio
import nest_asyncio
import signal
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Apply nest_asyncio to fix event loop issues
nest_asyncio.apply()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Initialize logger
logger = logging.getLogger(__name__)

class TinyLlamaBot:
    def __init__(self, token):
        self.token = token
        self.model = None
        self.tokenizer = None
        self.application = None
        self.shutdown_event = asyncio.Event()
        
    def initialize_model(self):
        """Initialize the TinyLlama model and tokenizer"""
        try:
            logger.info("Initializing TinyLlama model...")
            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                low_cpu_mem_usage=True,
                use_cache=True
            )
            logger.info("Model initialization complete")
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise

    async def generate_response(self, prompt: str) -> str:
        """Generate response using TinyLlama"""
        try:
            formatted_prompt = f"<human>: {prompt}\n<assistant>:"
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
            inputs = inputs.to(self.model.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split("<assistant>:")[-1].strip()
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while processing your request."

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handler for /start command"""
        welcome_message = "Hi! I'm an AI assistant powered by TinyLlama. How can I help you today?"
        await update.message.reply_text(welcome_message)

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handler for text messages"""
        user_message = update.message.text
        response = await self.generate_response(user_message)
        await update.message.reply_text(response)

    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Error handler"""
        logger.error(f"Error: {context.error} caused by {update}")
        if update:
            await update.message.reply_text("Sorry, I encountered an error processing your request.")

    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("Received shutdown signal")
        self.shutdown_event.set()

    async def shutdown(self):
        """Shutdown the bot gracefully"""
        logger.info("Shutting down...")
        if self.application:
            await self.application.stop()
            await self.application.shutdown()
        logger.info("Shutdown complete")

    def run(self):
        """Run the bot"""
        try:
            # Set up signal handlers
            signal.signal(signal.SIGINT, self.signal_handler)
            signal.signal(signal.SIGTERM, self.signal_handler)
            
            # Initialize the model
            self.initialize_model()
            
            # Create application and add handlers
            self.application = Application.builder().token(self.token).build()
            
            # Add handlers
            self.application.add_handler(CommandHandler("start", self.start))
            self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
            self.application.add_error_handler(self.error_handler)
            
            # Start the bot
            logger.info("Starting bot...")
            
            # Run the bot until shutdown is requested
            self.application.run_polling(
                allowed_updates=Update.ALL_TYPES,
                stop_signals=(),  # Disable default signal handling
                close_loop=False
            )
            
        except Exception as e:
            logger.error(f"Error running bot: {e}")
            raise
        finally:
            # Ensure cleanup happens
            asyncio.run(self.shutdown())

def main():
    # Replace YOUR_TOKEN with your actual Telegram bot token
    API_TOKEN = "7851332817:AAGPKkdSNzwqAYlxsvVPGcMoX-qqLOaoNpM"
    
    # Create and run bot
    bot = TinyLlamaBot(API_TOKEN)
    bot.run()

if __name__ == '__main__':
    main()