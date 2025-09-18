# General discord and python library imports
import os
import discord
from discord.ext import commands
from dotenv import load_dotenv, find_dotenv

# ChatGPT library imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter   # or RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import HumanMessage


# Load environment variables from .env file
load_dotenv(find_dotenv())

loader = TextLoader('./data.txt', encoding='utf-8')
documents = loader.load()
text_splitter = CharacterTextSplitter(
    separator="\n---\n",
    chunk_size=1200,
    chunk_overlap=50
)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
retriever = Chroma.from_documents(texts, embeddings).as_retriever()
chat = ChatOpenAI(temperature=0)



# This is your chatbots prompt template. Everytime someone asks a question in the chat, the AI will read this prompt before responding. 
promt_template = """--> YOUR-PROMPT <-- 
Example: 
You are the official Discord server help bot. You will only answer questions related to this server and its growing community.

Rules:
1. Rule 1
2. Rule 2
3. Rule 3
...
...
...

{context}

Please provide a concise and accurate answer to the user's question based on the above guidelines.

"""

prompt = PromptTemplate(
    template=promt_template, input_variables=["context"]
)
system_message_prompt = SystemMessagePromptTemplate(prompt=prompt)



# Set up Discord bot with intents
intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="", intents=intents) # Currently the bot responds to all messages sent in the targeted channel. You can change this in to your desired command --> command_prefix="!"


# Only respond to messages in a specific channel
TARGET_CHANNEL_ID = 1411802805995569162  # Replace with your channel's ID


# Define a command to handle questions
@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return
    if message.channel.id != TARGET_CHANNEL_ID:
        return

    try:
        question = message.content

        # Retrieve FAQ docs
        docs = retriever.get_relevant_documents(query=question)
        ctx = "\n\n".join(getattr(d, "page_content", str(d)) for d in docs[:4])[:3000]

        # LLM call
        sys = system_message_prompt.format(context=ctx)
        result = chat([sys, HumanMessage(content=question)])
        answer = result.content if hasattr(result, "content") else str(result)

        
        # Build embed (for a cleaner look)
        embed = discord.Embed(
            title="AI — Instant Support", # <-- Change this title to your prefered embed title for each message
            description=answer[:4096],
            color=discord.Color.blue(),
        )
        embed.set_footer(text=f"Asked by: {message.author.display_name}")
        embed.timestamp = message.created_at

        await message.channel.send(embed=embed)

    # If the user-message is formatted incorrectly, it will use this error handler
    except Exception as e:
        print(f"Error: {e}")
        await message.channel.send(
            embed=discord.Embed(
                title="Error",
                description="Sorry, I was unable to process your question.",
                color=discord.Color.red(),
            )
        )


# Run the bot with the token from environment variables

token = os.environ.get('DISCORD_TOKEN')
if not token:
    raise ValueError("DISCORD_TOKEN environment variable not set.")

bot.run(token)


