## 2018-08-14
1.crawel lyric_words program
3.handle rap-data program
2.crawel topic_words program
## 2018-08-13     
1.Asked for sun and lu et al. about the seq2seq program                     
2.After so much time,I understand almost what every function is doing in the program,but some details remained unknown.
## 2018-08-12         
None
## 2018-08-11      
None
## 2018-08-10
1. build test environment
- installed cuda9.0+cudnn7.1+tensorflow-gpu1.9 on my laboratory's desktop computer whose Graphics cards model is GTX1050
- run the program in my card,i found that the batch_size can't be set too large,otherwise, it will crash
- the card is not very good,but it's enough anyway.
2. have a visit to meituandianping
- met Mr Xie,who is in my same laboratory but a year ahead and worked there
- i didn't want to listen to the speech,how about you?
## 2018-08-09
1. seq2seq1.0
- data_helper.py
  - class Batch:encoder_inputs encoder_inputs_length decoder_targets decoder_targets_length         
  - def load_and_cut_data(filepath):return data
  - def create_dic_and_map(sources, targets):return sources_data, targets_data, word_to_id, id_to_word
  - def createBatch(sources, targets):return batch        
  - def getBatches(sources_data, targets_data, batch_size):return batches         
  - def sentence2enco(sentence, word2id): return batch
- model.py
  - class Seq2SeqModel(object)
    - def __init__(self, sess, rnn_size, num_layers, embedding_size, learning_rate, word_to_id, mode, use_attention,beam_search, beam_size, cell_type, max_gradient_norm, teacher_forcing, teacher_forcing_probability):
    - def build_graph(self)
      - def build_placeholder(self):
      - def build_encoder(self):
        - tf.nn.dynamic_rnn:
          - tf.nn.embedding_lookup
          - create_rnn_cell()
            - MultiRNNCell
              - single_rnn_cell
                - DropoutWrapper
                  - GRUCell(self.rnn_size) if self.cell_type == 'GRU' else LSTMCell(self.rnn_size)
      - def build_decoder(self):
        - build_train_decoder()
          - build_optimizer()
            - optimizer.apply_gradients
              - tf.clip_by_global_norm
              - tf.gradients
              - tf.trainable_variables
              - tf.train.AdamOptimizer
            - sequence_loss
              - tf.identity(decoder_outputs.rnn_output)
                - dynamic_decode
                  - BasicDecoder
                    - ScheduledEmbeddingTrainingHelper if self.teacher_forcing else TrainingHelper
                      - tf.nn.embedding_lookup
                        - tf.concat
                          - tf.strided_slice()
        - build_predict_decoder()
          - decoder_outputs.predicted_ids if beam_search else tf.expand_dims(decoder_outputs.sample_id, -1)
            - dynamic_decode(decoder=inference_decoder, maximum_iterations=50)
              - BasicDecoder+GreedyEmbeddingHelper if beam_search else BeamSearchDecoder
        - output_layer = tf.layers.Dense
        - build_decoder_cell()
          - decoder_cell.zero_state
            - batch_size = self.batch_size if not self.beam_search else self.batch_size * self.beam_size
            - AttentionWrapper
              - create_rnn_cell
              - BahdanauAttention
                - tile_batch if self.beam_search
                - nest.map_structure if self.beam_search     
    - def train(self, batch):
      - _, loss, summary = self.sess.run([self.train_op, self.loss, self.summary_op], feed_dict=feed_dict)
    - def eval(self, batch):
      loss, summary = self.sess.run([self.loss, self.summary_op], feed_dict=feed_dict)
    - def infer(self, batch):
      - predict = self.sess.run(self.decoder_predict_decode, feed_dict=feed_dict)          
  - train.py
    - model.train(nextBatch)
      - getBatches
        - model.saver.restore 
          - tf.train.get_checkpoint_state
            -  model = Seq2SeqModel()
        - create_dic_and_map
          - load_and_cut_data
  - predict.py
    - predict_ids_to_seq
      - model.infer(batch)
        - sentence2enco
          - sentence = sys.stdin.readline()
            - model.saver.restore 
              - tf.train.get_checkpoint_state
                -  model = Seq2SeqModel()
          - word_to_id=create_dic_and_map
            - load_and_cut_data

## 2018-08-08
1. Learn TensorFlow on this [website](https://www.bilibili.com/video/av20542427/?p=4)
- graphs/session/tensor/variable
- a logistic regression's example
- loss
- optimazer
- word2vec embedding
2. Learn Tensorboard            
I had a test in my computer, when i did all the procedure, everything is ok except the final page can't be opened.After searching for a while, i found many misleading solutions,i tried it,but it didn't work.i realized that my program's path comprises chinese, then it worked.
3. Count the fishes and ducks               
Walking around the weiming lake in peking university(yi lu shi pang wang,in order of age) after lunch, wang told many knowledge about the big ancient bronze bell to us,reaping the endless applause.
## 2018-08-07    
1. paper ' Chinese Poetry Generation with Planning based Neural Network ' 
- Introduction: this paper proposes a poetry generating method which generates poems in a two-stage procedure:plans the sub-topic of each line, generate the poem line by line.
- Keyword Extraction:TextRank algorithm
- Keyword Expansion: RNNLM-based method,Knowledge-based method.
- Poem Generation: the input consists of keywords and the previously generated text of the poem
- Dateset:　quatrains corpus, baidu baike and wikipedia
- Evaluation Metrics: for BLEU and METEOR have little correlation with human evaluation,use human evaluators
- Baselines: SMT/RNNLM/RNNPG/ANMT are implimented with the same pre-processing method.
- Results: the author's method outperforms all baseline models
- Automatic Generation vs. Human Poet: the second kind of experiments
- Conclusion: Keyword expansion method and keyword expansion method;the modified RNN model. 
2. paper 'Skip-Thought Vectors'
- Introduction: inspired by word2vec's skip-gram model, this paper encode a sentence to predict the sentences around it.The model is called skip-thoughts and vectors induced by the model are called skip-thought vectors.
- Inducing skip-thought vectors: 
  - Encoder: The last hidden state hNi represents the full sentence
  - Decoder: Separate parameters are used for each decoder with the exception of the vocabulary matrix V
  - Objective: the sum of the log-probabilities for the forward and backward sentences conditioned on the encoder representation
- Vocabulary expansion:　For words not seen during training, constract a matrix W to map Vw2v-Vrnn
- Experiments: evaluate skip-thoughts as a general feature extractor by reporting 8 tasks
- Conclusion: the sentence encoder is unsupervised,generic and distributed; a vocabulary expansion method to encode words is introduced.
## 2018-08-06    
1. paper ' Hafez: an Interactive Poetry Generation System '    
- Introduction: this paper is based on author's previous poetry generation system called Hafez. please click [this website](http://52.24.230.241/poem/advance/),you will know what it is.Try to use the system by adjusting any style configurations.
- Style Control: eight features are designed to control the style,that is,Encourage*discourage words/Curse words/repetition/alliteration/word length/topical words/sentiment/concrete words
- Speedup: Pre-load all parameters into RAM/Pre-calculate the rhyme types for all words/Shrink V’/beam search in GPU
- Learn a New Style Configuration: fit a quadratic regression between the rating change r and each weight change w independently
- Human-Computer Collaboration: this experiment verifies human collaboration can help Hafez generate better poems
- Automatic tuning for quality: reset each weight to the maximum of each quadratic curve to generate better default poems          
2. paper ' Generating Topical Poetry ' 
- Introduction: this paper introduce the Hafez1.0,a program that generates any number of distinct poems on a user-supplied topic.
- Vocabulary: arrange words to form a sequence of ten syllables alternating between stressed and unstressed
- Topically Related Words and Phrases: WordNet/PMI/word2vec,choose Word2vec to build vectors for phrases as well as words
- Choosing Rhyme Words: we first hash all related words/phrases into rhyme classes,then choose rhyme pairs randomly with probability
proportional to their score cosine(s,topical)
- Constructing FSA of Possible Poems: in the resulting FSA, each path is formally a sonnet.
- Path extraction through FSA with RNN: to locate fluent paths, we need a scoring function(train a two-layer LSTM) and a search procedure(beam search),we apply a penalty to repeating words,we apply a reward on all topically words,we train an encoder-decoder seq2seq model to further encourage the system to follow the topic
- Results and Analysis: We find that they generally stay on topic and are fairly creative
- Other Languages and Formats: generate Spanish-language poetry to show the generality of the approach
## 2018-08-05
Go out
## 2018-08-04
1. Seq2Seq的参数集合   
超参数：  
口层 数 = n ， hidden size = d   
口vocab for F =Vf ，vocab for E=Ve    
Encoder ：     
口 Input: input embedding for f ：Vf * d    
口 LSTM:第 一 层 ， 第 二 层 ： n(8d*d + 4d)      
Decoder:              
口 Input input embedding for e: Ve * d               
口 LSTM 第 一 层 ， 第 二 层 : n(8d*d + 4d)            
口 Output:                
        output embedding for e: Ve*d             
        output  bias for e: Ve                              
2. 与孙石闲聊                         
为什么来deecamp？学习一点机器学习深度学习                    
为什么来歌词组？主要标题吸引，其实应该是来打酱油的                 

## 2018-08-03     
理解逻辑，剖析原因       
1. 为什么需要seq2seq?        
机器翻译RNN（many to many）解决不了           
2. 为什么需要attention？             
翻译过程中注意力不一样，歌词生成中押韵              
3. 为什么需要BeamSearch？               
歌词生成中重复‘我爱我爱我’                  
4. 为什么需要bleu?              
歌词生成中语句通顺                  
5. 为什么需要分词？          
词向量，词的就不是单独的一个字           

## 2018-08-02
1. seq2seq程序（庞）进行分析，同时和‘吴亦凡’一起将整个程序的流程走了一遍         
  发现几点：1.这个程序和昨天的那个char-rnn程序极其相似，但是代码采用的是结构化文件，属于tensorflow模型代码，很棒             
           2.四部分分成四个文件，十分清晰，data_generator、model、train、predict              
           3.发现此程序的输出那里loss、summary里面的summary没有拿去生成tensorboard（陆伊），即么有log保存                
2. 交叉熵的概念还不是特别懂，看了一下李宏毅的goodness of function                 
3. 陆给的那个seq2seq代码试了一下，里面有一个jieba库和其他库，可能以后会用到                  
4. 回顾了一下防止过拟合：                 
  增加数据                  
  正则化（后面加了一个w平方的平均，不断迭代让部分w为0）                      
  dropout（一开始就让部分w为0）                          
5. 回顾了一下优化器                 
  SGD                 
  Momentum             
  NAG           
  Adagrad             
  RMSprop              
  Adadelta                   
  Adam                   

## 2018-08-01
1. seq2seq基本概念的学习          
2. attention-based model基本概念的学习           
3. char-rnn-cn的例子下载后进行测试            
  rnn_cell代码部分小改后            
  training：正常               
  Sampling：异常                     
            出现错误百度后说是作用域，但其实代码上有作用域。               
            最后重新启动TensorFlow正常输出。               
4. char-rnn-cn的例子代码进行学习重写                       



