import discord
from discord.ext import commands
import data_utils
import midi2audio
import keras


class Jazz(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command(aliases=["gen", "jazz", "music"])
    async def generate(self, ctx: commands.Context):
        await ctx.send("Composing...")
        data_utils.generate_music(self.bot.imodel)
        midi2audio.FluidSynth().midi_to_audio("music.midi", "music.flac")
        await ctx.send(file=discord.File("music.flac"))
        print("Done making music")


def setup(bot):
    bot.add_cog(Jazz(bot))
