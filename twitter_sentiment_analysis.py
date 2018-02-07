from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import time
import json
import sentiment_mod as s

#consumer key, consumer secret, access token, access secret.
ckey="OHSWiLZWApbtfERhITz6XQOik"
csecret="sbDcIsjBKZ0JHQvZBGZDOeOFy8Jp7wn8EpuvTkWHW7zN21neaQ"
atoken="960085083519684608-DIKmwQPzW20xrHGp8kahBy0MPASeqbS"
asecret="MDTgFXsqIMRq54d7XjIR5gopBMQJ92vPrmsGPQkbwEX4z"

class listener(StreamListener):

    def on_data(self, data):
        all_data = json.loads(data)
        tweet = all_data["text"]
        sentiment_value, confidence = s.sentiment(tweet)
        print(tweet,sentiment_value, confidence*100)

        if(confidence*100 >= 80):
            output = open("twitter-out.txt", "a")
            output.write(sentiment_value)
            output.write('\n')
            output.close()

        return True

    def on_error(self, status):
        print(status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["modi"])