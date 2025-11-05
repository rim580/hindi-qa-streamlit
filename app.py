from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

text = ("च्ची दोस्ती वही है जो मुश्किल वक्त में साथ दे। यह कहानी दो दोस्तों की गहरी दोस्ती की है। राहुल और सोहन एक ही गाँव में रहते थे। दोनों बचपन से दोस्त थे। वो साथ में स्कूल जाते, साथ में खेलते और अपनी हर बात एक-दूसरे से बाँटते थे। राहुल शांत स्वभाव का था, जबकि सोहन हँसमुख और हिम्मती था। एक दिन गर्मियों की छुट्टियाँ थीं। दोनों दोस्त जंगल में खेलने गए। वहाँ एक पुराना कुआँ था, जो सूख चुका था। राहुल खेलते-खेलते उस कुएँ के पास गया और अचानक उसका पैर फिसल गया। वो कुएँ में गिर पड़ा।राहुल ने जोर से चिल्लाया, “सोहन, मुझे बचाओ! मैं यहाँ से निकल नहीं सकता।” सोहन घबरा गया, लेकिन उसने हिम्मत नहीं हारी। उसने आसपास देखा तो उसे एक पुरानी रस्सी दिखी, जो पास के पेड़ से बँधी थी। वो दौड़कर रस्सी ले आया और उसे कुएँ में डाला। उसने कहा, “राहुल, इसे पकड़ो और ऊपर चढ़ने की कोशिश करो।” राहुल ने रस्सी पकड़ी और सोहन ने ऊपर से पूरी ताकत लगाकर उसे खींचा। कई कोशिशों के बाद राहुल बाहर आ गया। उसकी साँसें तेज़ चल रही थीं। उसने सोहन को गले लगाया और कहा, “अगर तू न होता तो मैं आज मर जाता। तू मेरा सच्चा दोस्त है।” सोहन हँसा और बोला, “दोस्ती में धन्यवाद नहीं बोलते। तू मेरा भाई है।”उस घटना के बाद उनकी दोस्ती और मज़बूत हो गई। सालों बाद राहुल एक बड़ा व्यापारी बना। उसने सोहन को अपने व्यापार में पार्टनर बनाया। दोनों ने मिलकर बहुत नाम कमाया। एक बार गाँव में मेला लगा, तो राहुल ने सोहन को स्टेज पर बुलाया और कहा, “ये मेरा दोस्त है, जिसने मेरी जान बचाई। आज जो कुछ भी हूँ, उसमें इसका बहुत बड़ा हाथ है।” सोहन ने भी हँसकर कहा, “हमारी दोस्ती हमेशा ऐसी ही रहेगी।” गाँव वाले उनकी दोस्ती की तारीफ करते नहीं थकते थे।")
class HindiQASystem:
    def __init__(self,text,chunk_size=3):
      self.original_text = text
      self.chunk_size = chunk_size

      self.chunks = []           # Will store text chunks
      self.vectorizer = None     # Will store TF-IDF vectorizer
      self.tfidf_matrix = None

    
      self._process_text()

      
      self._create_index()

      print("Q&A System is ready!\n")

  
    def _process_text(self):
       #split text on barakhadi
      sentences = re.split(r'[।\?!]+', self.original_text)
       #remove spaces and empty strings     
      sentences = [s.strip() for s in sentences if s.strip()] 
      print(f"   Found {len(sentences)} sentences in the text")

       #creating chunks
      for i in range(0, len(sentences), self.chunk_size):

          chunk_sentences = sentences[i:i + self.chunk_size]
          chunk = ' '.join(chunk_sentences)
          if chunk:
              self.chunks.append(chunk)  
      print(f"   Created {len(self.chunks)} chunks")

    def _create_index(self):
      self.vectorizer = TfidfVectorizer(
              ngram_range=(1, 2),  # Use single words and word pairs
              min_df=1,            # Minimum document frequency
              max_df=0.8,          # Maximum document frequency (ignore very common words)
              sublinear_tf=True    # Use logarithmic term frequency
          )
      self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)
          
      print(f"   Created vectors with {self.tfidf_matrix.shape[1]} unique features")
      print(f"   Matrix shape: {self.tfidf_matrix.shape}")

    def search(self,query,top_k=3):
      query_vector = self.vectorizer.transform([query])
          
          # Calculate cosine similarity between query and all chunks
          # Cosine similarity ranges from 0 (completely different) to 1 (identical)
      similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
          
          # Find the top_k most similar chunks
          # argsort gives indices, [-top_k:] takes last k, [::-1] reverses order
      top_indices = np.argsort(similarities)[-top_k:][::-1]
          
          # Collect results
      results = []
      for idx in top_indices:
              if similarities[idx] > 0:  # Only include if there's some similarity
                  results.append((
                      self.chunks[idx],      # The chunk text
                      similarities[idx],      # Similarity score
                      idx                     # Chunk index
                  ))
          
      return results
      
    # ------------------------------------------------------------------------
    # STEP 6: Answer Question Function
    # Main function users will call
    # ------------------------------------------------------------------------
    
    def answer_question(self, question, top_k=3, show_scores=False):
        """
        Answer a question by retrieving relevant text chunks
        
        Parameters:
        -----------
        question : str
            The question in Hindi
        top_k : int
            Number of chunks to retrieve
        show_scores : bool
            Whether to show similarity scores
            
        Returns:
        --------
        str : Formatted answer with relevant chunks
        """
        # Search for relevant chunks
        results = self.search(question, top_k)
        
        # If no results found
        if not results:
            return "क्षमा करें, मुझे इस प्रश्न का उत्तर नहीं मिला।"
        
        # Format the answer
        answer = f"प्रश्न: {question}\n\n"
        answer += "उत्तर (संबंधित जानकारी):\n"
        answer += "=" * 60 + "\n\n"
        
        for i, (chunk, score, idx) in enumerate(results, 1):
            answer += f"{i}. {chunk}\n"
            if show_scores:
                answer += f"   📊 समानता स्कोर: {score:.3f}\n"
            answer += "\n"
        
        return answer
    
    # ------------------------------------------------------------------------
    # STEP 7: Display System Info (Optional but useful)
    # ------------------------------------------------------------------------
    
    def get_system_info(self):
        """
        Display information about the Q&A system
        """
        info = "=" * 60 + "\n"
        info += "📚 Hindi Q&A System Information\n"
        info += "=" * 60 + "\n"
        info += f"Total text length: {len(self.original_text)} characters\n"
        info += f"Chunk size: {self.chunk_size} sentences\n"
        info += f"Number of chunks: {len(self.chunks)}\n"
        info += f"Vocabulary size: {len(self.vectorizer.vocabulary_)} unique words\n"
        info += "=" * 60
        return info


# ============================================================================
# STEP 8: Example Usage
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("हिंदी प्रश्न-उत्तर प्रणाली - Complete Build")
    print("="*70 + "\n")
    
   
    
    # Create the Q&A system
    qa_system = HindiQASystem(text, chunk_size=2)
    
    # Display system information
    print(qa_system.get_system_info())
    
    # Ask some questions
    print("\n" + "="*70)
    print("प्रश्नों के उत्तर:")
    print("="*70 + "\n")
    
    questions = ["कहानी के दो मुख्य पात्र कौन हैं?,राहुल और सोहन कहाँ रहते थे?,राहुल का स्वभाव कैसा था?"]
    
    for question in questions:
        answer = qa_system.answer_question(question, top_k=2, show_scores=True)
        print(answer)
        print("-"*70 + "\n")
    

