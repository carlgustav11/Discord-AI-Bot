# HOW TO USE THIS BOT

This AI chat-bot is easy to configure and even easier to use.

# Requirements:
- Selfhosted python bot
- Administrator or owner of a Discord channel
- OpenAI account (to get an API KEY) 
  OBS!! API use from OpenAI has a payment per Mil token


## -- How to setup the bot --

### Getting the neccesary keys:
1. Get you OpenAI API KEY 
   ```(openai.com)```
2. Insert this key in the .env file
3. Create a Discord bot from 
   ```(discord.com/developers/applications)```
4. Give your bot, message and read -authorization (or give it admin privilages)
5. Get you bot Token and insert it in the .env file
6. Use the generated URL to invite your bot to your channel

### Configure the bot:
7. Choose what channel you want the bot to operate in, and get the channel ID by right-clicking on the channel and copy channel ID (when in developer mode on Discord)
   Paste this channel ID in the ```TARGET_CHANNEL_ID = YOUR-CHANNEL-ID``` variable
8. By default, the bot is going to respond to every message asked by any user with channel access. 
   You can change this by configuring the ```command_prefix``` and change it from blank to ```"!" or "%"``` or any desired command.
9. To add a title to the top of each message, change the title variable in:
   ```#Build embed
   embed = discord.Embed(
        title="YOUR-PREFERED-TITLE"```

### Bot prompt:
10. To prompt your bot, use the ```prompt_template = """YOUR-PROMPT"""```
    to make sure your bot answers the users in the way you want it to.
    This works like any other AI prompt, and it can be as long and specific as you want.
11. Use the data.txt file to make pre-made answers that the bot will use, in case a user asks a question that matches an answer from the data.txt file.
    The bot will always prioritize the pre-made answers, and if it doesn't exist, it will generate an answer from the main prompt
12. Run ```pip install discord.py python-dotenv langchain langchain-community langchain-openai``` in you terminal.
    Else do them one at a time for less risk of syntax error
    ```pip install discord.py
       pip install python-dotenv 
       pip install langchain 
       pip install langchain-community 
       pip install langchain-openai```

