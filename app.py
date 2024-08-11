import os.path
import pandas as pd
import matplotlib.pyplot as plt
import re
import streamlit as st
import streamlit.components.v1 as stc
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import base64
from bs4 import BeautifulSoup
from wordcloud import WordCloud
from openai import OpenAI


# Header
html_temp = """
		<div style="background-color:#B7CDB7; padding:10px; border-radius:10px">
		<h1 style="color:white; text-align:center; font-family:'Garamond', serif; ">Streamlit Application for Gmail Analysis</h1>
		<h4 style="color:white; text-align:center; font-family:'Garamond', serif; ">Word Cloud, Email Summary, and Sentiment Analysis</h4>
		</div>
		"""


def gmail_access():
	"""
	Funtion to authenticate and connect Gmail account to the application.
	"""

	# If modifying these scopes, delete the file token.json.
	SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

	creds = None
	# The file token.json stores the user's access and refresh tokens, and is created automatically 
	# when the authorization flow completes for the first time.
	if os.path.exists("token.json"):
		creds = Credentials.from_authorized_user_file("token.json", SCOPES)
	
	# If there are no (valid) credentials available, let the user log in.
	if not creds or not creds.valid:
		if creds and creds.expired and creds.refresh_token:
			creds.refresh(Request())
		else:
			flow = InstalledAppFlow.from_client_config({
				"installed": {
		                        "client_id": st.secrets["gmail"]["client_id"],
		                        "project_id": st.secrets["gmail"]["project_id"],
		                        "auth_uri": st.secrets["gmail"]["auth_uri"],
		                        "token_uri": st.secrets["gmail"]["token_uri"],
		                        "auth_provider_x509_cert_url": st.secrets["gmail"]["auth_provider_x509_cert_url"],
		                        "client_secret": st.secrets["gmail"]["client_secret"],
		                        "redirect_uris": ["http://localhost"]
				}},SCOPES)
			auth_url, _ = flow.authorization_url(prompt='consent')
			# creds = flow.run_local_server(port=0)
			st.write("Go to this URL to authorize the application:")
			st.write(auth_url)
			code = st.text_input('Enter the authorization code here:')
			if code:
			        flow.fetch_token(code=code)
			        creds = flow.credentials
		
		# Save the credentials for the next run
		# with open("token.json", "w") as token:
		#   token.write(creds.to_json())

	# Connect to Gmail API
	service = build("gmail", "v1", credentials=creds)
	return service


def retrieve_spam(service):
	"""
	Function to retrieve all emails in Spam folder and return subject and sender of email.
	"""

	try:
		results = service.users().messages().list(userId="me", labelIds=["SPAM"], maxResults=30).execute() 
		messages = results.get("messages", [])

		if not messages:
			return
		
		else:
			email_content = []
			for message in messages:
				email = service.users().messages().get(userId="me", id=message["id"]).execute()
				
				# Extract email headers to get subject and sender
				headers = email["payload"]["headers"]
				subject = next(header['value'] for header in headers if header['name'] == 'Subject')
				sender = next(header['value'] for header in headers if header['name'] == 'From')

				# Extract email body
				parts = email["payload"].get('parts', [])
				body = ""
				if not parts: # non-multipart emails
					body = email['payload'].get('body', {}).get('data', '')
				else: # multi-part emails
					for part in parts:
						mime_type = part.get("mimeType")
						data = part.get("body", {}).get("data", "")

						if mime_type == "text/plain":
							body += base64.urlsafe_b64decode(data).decode('utf-8')
						elif mime_type == "text/html":
							html_body = base64.urlsafe_b64decode(data).decode('utf-8')
							soup = BeautifulSoup(html_body, "html.parser")
							body += soup.get_text()
				body = re.sub(r'http\S+|www\S+|https\S+|\r?\n+|[^\w\s]', '', body) # exclude URLs, new lines, non-words

				email_content.append({"Subject": subject, "Sender": sender, 'Content': body})

			df = pd.DataFrame(email_content, columns=["Subject", "Sender", "Content"])
			return df

	except HttpError as error:
		# TODO(developer) - Handle errors from gmail API.
		print(f"An error occurred: {error}")


def generate_wordcloud(df):
	"""
	Function to word cloud from 30 latest emails.
	"""

	text = ' '.join(df["Content"].astype(str))
	wordcloud = WordCloud(max_words=50, background_color="white", width=800, height=400).generate(text)
	fig, ax = plt.subplots(figsize=(20,10))
	ax.imshow(wordcloud, interpolation="bilinear")
	ax.axis("off")
	st.pyplot(fig)


def email_summary(email_selection, df):
	"""
	Function to show summary of selected email using GPT API.
	"""
	
	selected_email = df.loc[email_selection]
	selected_body = selected_email["Content"]

	client = OpenAI(api_key=st.secrets["openai"]["api_key"])

	def generate_summary(prompt):
		response = client.chat.completions.create(
			model = "gpt-4o-mini",
			messages = [
				{
				"role": "system",
				"content": "A block of text will be provided to you and your task is to provide a summary of this text."
				},
				{
				"role": "user",
				"content": f"Summarize the following text: {prompt}"
				}
			],
			temperature = 0.5,
			max_tokens = 256)
		return response.choices[0].message.content.strip()
	
	st.write(generate_summary(selected_body))


def sentiment_analysis(email_selection, df):
	"""
	Function to show sentiment analysis of selected email using GPT API.
	"""
	
	selected_email = df.loc[email_selection]
	selected_body = selected_email["Content"]

	client = OpenAI(api_key=st.secrets["openai"]["api_key"])

	def generate_sentiment(prompt):
		response = client.chat.completions.create(
			model = "gpt-4o-mini",
			messages = [
				{"role": "system",
				"content": "A block of text will be provided to you and your task is to assess the sentiment of this text. Return one word based on the sentiment: Positive, Negative, or Neutral."
				},
				{"role": "user",
				"content": "The weather was horrendous."
				},
				{"role": "assistant",
				"content": "Negative"
				},
				{"role": "user",
				"content": "It was a pleasant experience."
				},
				{"role": "assistant",
				"content": "Positive"
				},
				{"role": "user",
				"content": "The professor is great but the exams were hard."
				},
				{"role": "assistant",
				"content": "Neutral"
				},
				{"role": "user",
				"content": f"Provide a sentiment analysis of the following text: {prompt}"
				}
			],
			temperature = 0.5,
			max_tokens = 256)
		return response.choices[0].message.content.strip()
	
	st.write(generate_sentiment(selected_body))


def main():
	stc.html(html_temp)
	st.markdown("This application allows users to log in to their Gmail accounts and access the emails from the Spam folder. It offers the following functionalities: word cloud generation, email summary, and sentiment analysis of the Spam emails.")
	st.text(" ")

	# Sidebar
	st.sidebar.title("Gmail Analysis")
	choice = st.sidebar.selectbox("Options", ["Home", "Word Cloud Generation", "Email Summary and Sentiment Analysis"])

	if choice == "Home":
		st.subheader("Gmail Access")

		# Initialize session state for authentication status
		if "authenticated" not in st.session_state:
			st.session_state.authenticated = False
			st.session_state.service = None

		# Authenticate and connect to Gmail account
		if st.button("Allow access to Gmail account"):
			st.session_state.service = gmail_access()
			st.session_state.authenticated = True
			st.write("Gmail account access granted.")
		
		# Retrieve and display Spam emails
		if st.session_state.authenticated:
			if st.button("Show Spam emails"):
				if st.session_state.service:
					df = retrieve_spam(st.session_state.service)
					if df is not None:
						if not df.empty:
							st.write("###### Spam (latest 30 emails):")
							st.write(df[["Subject", "Sender"]].to_html(index=False), unsafe_allow_html=True)
					else:
						st.write("No Spam emails to display.")
				else:
					st.write("Gmail not initialized. Authenticate first.")
		else:
			st.write("Authenticate first to access Spam emails.")
		
	elif choice == "Word Cloud Generation":
		st.subheader("Word Cloud Generation")
		if st.session_state.authenticated:
			if st.button("Generate word cloud"):
				if st.session_state.service:
					df = retrieve_spam(st.session_state.service)
					if df is not None:
						if not df.empty:
							generate_wordcloud(df)
					else:
						st.write("No Spam emails to display.")
		else:
			st.write("Authenticate first to access Spam emails.")
		

	elif choice == "Email Summary and Sentiment Analysis":
		if st.session_state.authenticated:
			if st.session_state.service:
				st.subheader("Latest 10 Emails")
				df = retrieve_spam(st.session_state.service)
				if df is not None:
					df = df.loc[:9]
					df.index +=1
					st.write(df[["Subject", "Sender"]].to_html(), unsafe_allow_html=True)
					st.write(" ")

					st.subheader("Email Selection")
					email_selection = st.selectbox("Select an email", df.index, format_func=lambda x: df.loc[x, 'Subject'])

					st.subheader("Email Summary")
					email_summary(email_selection, df)

					st.subheader("Sentiment Analysis")
					sentiment_analysis(email_selection, df)
				else:
					st.write("No Spam emails to display.")
		else:
			st.write("Authenticate first to access spam emails.")

if __name__ == '__main__':
	main()
