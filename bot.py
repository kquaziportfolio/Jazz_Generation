import os
import discord
from discord.ext.commands import *
from jishaku.help_command import *
import keras
import music_utils  # Lamba layer uses it
print("Loading Discord Bot")
intents = discord.Intents.all()

TRAIN_MODEL_ON_LOAD = True  # train model on load
if TRAIN_MODEL_ON_LOAD:
    import jazz
    imodel = jazz.imodel
else:
    imodel = keras.models.load_model("imodel")
print("Model loaded")


class Take5Bot(Bot):
    def __init__(self, *args, prefix=None, **kwargs):
        super().__init__(prefix, *args, **kwargs)
        self.imodel = imodel
        self.invite = None

    async def on_ready(self):
        status = "Vibing at a Juice World concert (⌐■_■)"
        self.invite = discord.utils.oauth_url(
            self.user.id, permissions=discord.Permissions(permissions=8)
        )
        print(
            "Logged in as",
            client.user.name,
            "\nId:",
            client.user.id,
            "\nOath:",
            self.invite,
        )
        print("--------")
        await self.change_presence(
            activity=discord.Game(name=status), status=discord.Status.online
        )

    async def on_message(self, message: discord.Message):
        ctx = await self.get_context(message)
        try:
            await self.process_commands(message)
        except Exception as ex:
            print(ex)

    async def logout(self):
        await super().logout()

    async def on_command_error(self, ctx, error):
        await ctx.send(error)

    async def process_commands(self, message):
        await super().process_commands(message)


if __name__ == "__main__":
    client = Take5Bot(
        intents=intents,
        prefix=when_mentioned_or("j!"),
        help_command=DefaultPaginatorHelp(),
    )
    for file in os.listdir("cogs"):
        if file.endswith(".py"):
            name = file[:-3]
            try:
                client.load_extension(f"cogs.{name}")
            except Exception as e:
                print(f"Failed to load cog {name} due to error\n", e)
    client.load_extension("jishaku")
    try:
        client.run(input("Token: "))
    except Exception as e:
        print("Bai")
        raise
