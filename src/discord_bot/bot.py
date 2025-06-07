"""
Discord Bot with TTS Streaming Capabilities
Integrates real-time speech-to-text and text-to-speech for Discord voice channels
"""

import asyncio
import logging
import io
from typing import Optional, Dict, Any
import discord
from discord.ext import commands
import numpy as np

from models.base_trainer import ModelRegistry, InferenceResult
from utils.logging_utils import get_logger


logger = get_logger(__name__)


class TTSBot(commands.Bot):
    """
    Discord bot with TTS streaming and voice cloning capabilities.
    
    Features:
    - Real-time text-to-speech in voice channels
    - Voice cloning from reference audio
    - Speech-to-text transcription
    - Multiple TTS model support (XTTS, VITS, Tortoise)
    """
    
    def __init__(self, model_type: str = "xtts_v2", voice_clone_path: Optional[str] = None):
        # Configure intents
        intents = discord.Intents.default()
        intents.message_content = True
        intents.voice_states = True
        
        super().__init__(command_prefix='!', intents=intents)
        
        self.model_type = model_type
        self.voice_clone_path = voice_clone_path
        self.tts_model = None
        self.is_listening = False
        self.voice_clients: Dict[int, discord.VoiceClient] = {}
        
        # Setup commands
        self._setup_commands()
    
    async def setup_hook(self):
        """Initialize the bot when it starts."""
        logger.info(f"Setting up TTS Bot with {self.model_type} model")
        
        try:
            # Load TTS model
            self.tts_model = ModelRegistry.get_trainer(self.model_type)
            logger.info(f"✅ TTS model {self.model_type} loaded successfully")
            
            # Load voice clone reference if provided
            if self.voice_clone_path:
                logger.info(f"🎤 Voice clone reference loaded: {self.voice_clone_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup TTS model: {e}")
            raise
    
    async def on_ready(self):
        """Called when the bot is ready."""
        logger.info(f"🤖 {self.user} is now online and ready!")
        logger.info(f"🔊 TTS Model: {self.model_type}")
        logger.info(f"🎮 Guilds: {len(self.guilds)}")
        
        # Set bot status
        activity = discord.Activity(type=discord.ActivityType.listening, name="!say <text>")
        await self.change_presence(activity=activity)
    
    async def on_message(self, message):
        """Handle incoming messages."""
        if message.author == self.user:
            return
        
        # Log messages for debugging
        if message.content.startswith('!'):
            logger.debug(f"Command from {message.author}: {message.content}")
        
        await self.process_commands(message)
    
    def _setup_commands(self):
        """Setup bot commands."""
        
        @self.command(name='say')
        async def say_command(ctx, *, text: str):
            """Make the bot speak text in the voice channel."""
            await self._handle_say_command(ctx, text)
        
        @self.command(name='join')
        async def join_command(ctx):
            """Join the user's voice channel."""
            await self._handle_join_command(ctx)
        
        @self.command(name='leave')
        async def leave_command(ctx):
            """Leave the voice channel."""
            await self._handle_leave_command(ctx)
        
        @self.command(name='clone')
        async def clone_command(ctx):
            """Clone voice from attached audio file."""
            await self._handle_clone_command(ctx)
        
        @self.command(name='listen')
        async def listen_command(ctx):
            """Start listening and transcribing voice channel."""
            await self._handle_listen_command(ctx)
        
        @self.command(name='stop')
        async def stop_command(ctx):
            """Stop current TTS or listening."""
            await self._handle_stop_command(ctx)
        
        @self.command(name='model')
        async def model_command(ctx, model_type: str = None):
            """Switch TTS model or show current model."""
            await self._handle_model_command(ctx, model_type)
    
    async def _handle_say_command(self, ctx, text: str):
        """Handle the !say command."""
        try:
            # Check if user is in a voice channel
            if not ctx.author.voice:
                await ctx.send("❌ You need to be in a voice channel!")
                return
            
            # Join voice channel if not already connected
            voice_client = await self._ensure_voice_connection(ctx)
            if not voice_client:
                await ctx.send("❌ Failed to join voice channel!")
                return
            
            # Generate TTS audio
            await ctx.send(f"🎤 Generating speech: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            
            # Create temporary audio file
            output_path = f"temp_tts_{ctx.guild.id}.wav"
            
            result = await self.tts_model.synthesize(
                text=text,
                reference_audio=self.voice_clone_path,
                output_path=output_path,
                streaming=True
            )
            
            if not result.success:
                await ctx.send(f"❌ TTS generation failed: {result.error}")
                return
            
            # Play audio in voice channel
            if voice_client.is_playing():
                voice_client.stop()
            
            audio_source = discord.FFmpegPCMAudio(output_path)
            voice_client.play(audio_source, after=lambda e: logger.error(f"Player error: {e}") if e else None)
            
            await ctx.send(f"🔊 Speaking in {ctx.author.voice.channel.mention}")
            
        except Exception as e:
            logger.error(f"Say command error: {e}")
            await ctx.send(f"❌ Error: {str(e)}")
    
    async def _handle_join_command(self, ctx):
        """Handle the !join command."""
        try:
            if not ctx.author.voice:
                await ctx.send("❌ You need to be in a voice channel!")
                return
            
            channel = ctx.author.voice.channel
            voice_client = await channel.connect()
            self.voice_clients[ctx.guild.id] = voice_client
            
            await ctx.send(f"✅ Joined {channel.mention}")
            logger.info(f"Joined voice channel: {channel.name} in {ctx.guild.name}")
            
        except Exception as e:
            logger.error(f"Join command error: {e}")
            await ctx.send(f"❌ Failed to join: {str(e)}")
    
    async def _handle_leave_command(self, ctx):
        """Handle the !leave command."""
        try:
            voice_client = ctx.voice_client
            if voice_client:
                await voice_client.disconnect()
                if ctx.guild.id in self.voice_clients:
                    del self.voice_clients[ctx.guild.id]
                await ctx.send("👋 Left voice channel")
            else:
                await ctx.send("❌ Not connected to a voice channel!")
                
        except Exception as e:
            logger.error(f"Leave command error: {e}")
            await ctx.send(f"❌ Error leaving: {str(e)}")
    
    async def _handle_clone_command(self, ctx):
        """Handle the !clone command."""
        try:
            if not ctx.message.attachments:
                await ctx.send("❌ Please attach an audio file to clone!")
                return
            
            attachment = ctx.message.attachments[0]
            if not any(attachment.filename.lower().endswith(ext) for ext in ['.wav', '.mp3', '.flac', '.ogg']):
                await ctx.send("❌ Please attach a valid audio file (.wav, .mp3, .flac, .ogg)")
                return
            
            # Download audio file
            audio_path = f"temp_clone_{ctx.guild.id}_{attachment.filename}"
            await attachment.save(audio_path)
            
            # Set as voice clone reference
            self.voice_clone_path = audio_path
            
            await ctx.send(f"✅ Voice cloned from {attachment.filename}! Use `!say` to test it.")
            logger.info(f"Voice cloned from {attachment.filename} for guild {ctx.guild.name}")
            
        except Exception as e:
            logger.error(f"Clone command error: {e}")
            await ctx.send(f"❌ Voice cloning failed: {str(e)}")
    
    async def _handle_listen_command(self, ctx):
        """Handle the !listen command."""
        await ctx.send("🎧 Listening feature coming soon! Will transcribe voice channel audio.")
        # TODO: Implement voice channel listening and transcription
    
    async def _handle_stop_command(self, ctx):
        """Handle the !stop command."""
        try:
            voice_client = ctx.voice_client
            if voice_client and voice_client.is_playing():
                voice_client.stop()
                await ctx.send("⏹️ Stopped current audio")
            else:
                await ctx.send("❌ Nothing is currently playing!")
                
        except Exception as e:
            logger.error(f"Stop command error: {e}")
            await ctx.send(f"❌ Error stopping: {str(e)}")
    
    async def _handle_model_command(self, ctx, model_type: str = None):
        """Handle the !model command."""
        if not model_type:
            await ctx.send(f"🤖 Current model: **{self.model_type}**\nAvailable: {', '.join(ModelRegistry.list_available_models())}")
            return
        
        try:
            if model_type not in ModelRegistry.list_available_models():
                await ctx.send(f"❌ Unknown model: {model_type}\nAvailable: {', '.join(ModelRegistry.list_available_models())}")
                return
            
            # Switch model
            self.model_type = model_type
            self.tts_model = ModelRegistry.get_trainer(model_type)
            
            await ctx.send(f"✅ Switched to model: **{model_type}**")
            logger.info(f"Switched TTS model to {model_type} for guild {ctx.guild.name}")
            
        except Exception as e:
            logger.error(f"Model command error: {e}")
            await ctx.send(f"❌ Failed to switch model: {str(e)}")
    
    async def _ensure_voice_connection(self, ctx) -> Optional[discord.VoiceClient]:
        """Ensure bot is connected to user's voice channel."""
        if not ctx.author.voice:
            return None
        
        # If already connected to the same channel, return existing connection
        if ctx.voice_client and ctx.voice_client.channel == ctx.author.voice.channel:
            return ctx.voice_client
        
        # Connect to the user's channel
        try:
            voice_client = await ctx.author.voice.channel.connect()
            self.voice_clients[ctx.guild.id] = voice_client
            return voice_client
        except Exception as e:
            logger.error(f"Failed to connect to voice channel: {e}")
            return None
    
    async def cleanup_temp_files(self):
        """Cleanup temporary audio files."""
        import os
        import glob
        
        temp_files = glob.glob("temp_tts_*.wav") + glob.glob("temp_clone_*")
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
                logger.debug(f"Cleaned up temp file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {temp_file}: {e}")


async def run_bot(token: str, model_type: str = "xtts_v2", voice_clone_path: Optional[str] = None):
    """
    Run the Discord TTS bot.
    
    Args:
        token: Discord bot token
        model_type: TTS model to use
        voice_clone_path: Path to reference audio for voice cloning
    """
    bot = TTSBot(model_type=model_type, voice_clone_path=voice_clone_path)
    
    try:
        await bot.start(token)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot error: {e}")
    finally:
        await bot.cleanup_temp_files()
        await bot.close() 