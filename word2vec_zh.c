//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40


/*
 * 词典的哈希表的大小
 * 词汇量不能超过这个数字的70%
 * 例如，如果哈希表有30M个条目，那么词典的最大大小为21M。
 * 这是为了尽量减少散列碰撞的发生和性能影响。
 */
const int vocab_hash_size = 30000000;  // 最多存2000万的单词以保证查找效率

typedef float real;                    // float 作为浮点数的精度。指数位八位，尾数位23位。

/**
 * ======== 词 ========
 * Properties:
 *   cn - 词频(出现的次数)。
 *   word - 用字符串表示的词。
 */
struct vocab_word {
  long long cn;
  int *point;
  char *word, *code, codelen;
};

/*
 * ======== Global Variables ========
 *
 */
char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];

/*
 * ======== vocab ========
 *
 * ======== 词典 =========
 * 这个数组将保存词汇表中的所有单词。 
 *
 * 私有属性。
 */
struct vocab_word *vocab;

int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;

/*
 * ======== vocab_hash ========
 * 这个数组是词典的哈希表。词被映射成一个hashcode(一个整数)，
 * hashcode 作为 'vocab_hash' 的索引，以检索 'vocab' 数组中词的索引。
 */
int *vocab_hash;

/*
 * ======== vocab_max_size ========
 * 这不是对词典中单词数量的限制，而是用于分配词典表的块大小。
 * 词典将根据需要扩展，并分配，例如一次一千个单词。
 *
 * ======== vocab_size ========
 * 词典大小。
 * 私有属性。
 *
 * ======== layer1_size ========
 * 词向量维度。
 * 隐藏层的神经元个数。
 */
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;

/*
 *
 */
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0;

/*
 * ======== alpha ========
 * 学习率
 *
 * ======== starting_alpha ========
 *
 * ======== sample ========
 * 这个参数用于控制频繁词的下采样。
 * "sample" 的值较小意味着单词不太可能被留下。
 * 将 sample 设置为 0 取消下采样。
 * 更多细节参考下采样部分的注释。
 */
real alpha = 0.025, starting_alpha, sample = 1e-3;

/*
 * 重要 - 参数矩阵以一维数组存储，不是二维矩阵，
 * 所以要访问 syn0 的第i行，索引是 i * layer1_size。
 * 
 * ======== syn0 ========
 * hidden layer的参数。
 * 即词向量。
 *
 * ======== syn1 ========
 * 使用 HS 的 output 层参数。
 * 
 * ======== syn1neg ========
 * 使用 NS 的output 层参数。
 *
 * ======== expTable ========
 * 预先计算好的 sigmoid 值的表。
 */
real *syn0, *syn1, *syn1neg, *expTable;
clock_t start;

int hs = 0, negative = 5;
// 静态 unigram 采样表的规模
const int table_size = 1e8;
// unigram 采样表
int *table;

/**
 * ======== InitUnigramTable ========
 * 这个表用于实现 negative sampling。
 * 每个词的权重等于他的词频的(3/4)次方。
 * 选择一个词的概率就是他的权重除以所有单词的权重之和。
 *
 * 词典已经按照词频，降序排序。
 * 将按词频由小到大排列。
 */
void InitUnigramTable() {
  int a, i;
  double train_words_pow = 0;
  table = (int *)malloc(table_size * sizeof(int));
  double d1, power = 0.75;
  
  // 给表分配内存，它比词典更大，因为单词的词频不同，有的词会出现多次。
  // 每个词在表中至少出现一次。
  // table 的大小相对于词典的大小决定了采样率。
  // 更大的 unigram table 意味着将会更以接近公式计算的概率来选择负样本。
  // table 反应的是一个单词能量的分布，一个单词能量越大，所占位置越多。
  table = (int *)malloc(table_size * sizeof(int));
  
  // 计算分母，即所有词的权重总和。
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
 
  // 'i' 是当前单词的词汇索引，而 'a' 将是 unigram 表的索引。
  i = 0;
  
  // 计算词 'i' 的概率，(0,1]
  d1 = pow(vocab[i].cn, power) / train_words_pow;
  
  // 遍历整个表
  // a - unigram table 表的索引
  // i - 词典的索引
  for (a = 0; a < table_size; a++) {
    // 将单词索引存入 
    // 根据其词频，词i将在表里多次出现，次数取决于词频, 即能量越大，占的空间越多。
    table[a] = i;
    
    // a 是静态采样表当前遍历的索引，分母是静态采样表的大小，
    // d1 代表当前遍历到的所有词的能量总和。
    // 当满足一下条件，移动到下一个词.
    if (a / (double)table_size > d1) {
      // 下个词
      i++;
      
      // 计算新词的概率，并将其与之前所有单词的概率相加，
      // 这样我们就可以将d1与我们已填满部分所占百分比进行比较。
      d1 += pow(vocab[i].cn, power) / train_words_pow;
    }
    // 所有单词的权重总和应该是1，所以在表的末尾不应该有任何额外的空间。
    // 这里是预防措施，预防万一出现i超出词典长度的情况。
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

/**
 * ======== ReadWord ========
 * 从文件中读词，假设 space + tab + EOL 是词的边界。
 *
 * Parameters:
 *   word - 一个已经分配了最大字符串长度的 char 数组。
 *   fin  - 训练文件。
 */
void ReadWord(char *word, FILE *fin) {
  
  // a 是词的索引.
  int a = 0, ch;
  
  // 读到词的尾部或文件的尾部。
  // feof 检测流上的文件结束符。
  while (!feof(fin)) {
  
    // 获取下个字符。
    ch = fgetc(fin);
    
    // ASCII 里 13 代表回车CR,读取下个字符。
    if (ch == 13) continue;
    
    // 当遇到space(' ') + tab(\t) + EOL(\n)时，认为word结束。
    // UNIX/Linux中 '\n' 为一行的结束符号，
    // windows中为: "<回车><换行>" , 即 "\r\n"；
    // Mac系统里，每行结尾是"<回车>" , 即"\r"。
    // 检查词的边界。
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        // ungetc 将字符退回到输入流,所以我们下次还可以找的到。
        // 将 '\n' 退回到流中。
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      // 如果单词为空且字符为换行符,将其作为句子的结尾,并用标记 </s> 来标记它。
      // 代表句子结束。
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
        // 如果单词是空,且字符是 space 或 tab,继续遍历即可。
      } else continue;
    }
    
    // 如果字符不是 space, tab, CR, newline,  将其添加到词中。
    word[a] = ch;
    a++;
    
    // 如果词的长度太长，将其截断，但是要继续直到结束。
    if (a >= MAX_STRING - 1) a--;
  }
  
  // string 以 /0 终结。
  word[a] = 0;
}

/**
 * ======== GetWordHash ========
 * 返回一个词的hash 值。
 * 一一对应。
 * vocab_hash_size (default is 30E6).
 *
 * For example, the word 'hat':
 * hash = ((((h * 257) + a) * 257) + t) % 30E6
 * 冲突是可能的，采用开放地址发来解决。
 */
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

/**
 * ======== SearchVocab ========
 * 使用单词的hash查找单词的索引。
 * 如果不存在，返回-1。
 * 这个函数用于词的快速检索。
 * 如果词的数量过大 即 >> 30E6 ，这个函数可能会数组越界。
 * 需要提高 min_reduce 来处理。
 */
int SearchVocab(char *word) {
  // 计算 变量 word 的 hash值。
  unsigned int hash = GetWordHash(word);
  
  // 在散列表中查找索引，需要处理冲突。
  // 冲突处理在 AddWordToVocab 函数中。
  while (1) {
    // 如果不在散列表中,就不在词典中。
    if (vocab_hash[hash] == -1) return -1;
    
    // 验证输入的单词是否与所查找的词匹配。
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    
    // 开放地址法查询。
    hash = (hash + 1) % vocab_hash_size;
  }
  
  // 永远不会访问。
  return -1;
}

/**
 * ======== ReadWordIndex ========
 * 读取语料的下个单词，并返回在词典 vocab 中的索引。
 */
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

/**
 * ======== AddWordToVocab ========
 * 添加一个新单词到词典中，在词典中还不存在。
 */
int AddWordToVocab(char *word) {
  // 单词长度。
  unsigned int hash, length = strlen(word) + 1;
  
  // 限制单词长度。
  if (length > MAX_STRING) length = MAX_STRING;
  
  // 分配并存储词字符串。
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  
  // 初始化词频为0
  vocab[vocab_size].cn = 0;
  
  // 增加词典的长度。
  vocab_size++;
  
  // 给词典增加内存。
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  
  // 词的hash code, 在 0 到 30E6 之间。
  hash = GetWordHash(word);
  
  // 开放地址法。
  while (vocab_hash[hash] != -1) 
    hash = (hash + 1) % vocab_hash_size;
  
  // 将词在词典中的索引插入到散列表中。
  vocab_hash[hash] = vocab_size - 1;
  
  // 返回单词在词典中的索引。
  return vocab_size - 1;
}

// 比较器，用于按词频排序。
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

/**
 * ======== SortVocab ========
 * 按照词频排序，并移除语料中词频小于 min_reduce 的词。
 * 
 * 移除词需要重新计算散列表。
 */
void SortVocab() {
  int a, size;
  unsigned int hash;
  
  /**
   * 按照词频对词典进行降序排序。
   *
   * </s> 在第一个位置，从索引 1 开始排序。
   *
   * 以这种方式对词典进行排序，会使词频最少的单词出现在词典的尾部。
   * 这允许我们直接释放那些需要被移除的词相关的内存。
   * 非常好的技巧，可以省略掉重新分配。
   */
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  
  // 重新初始化散列表。
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  
  // 用于长度记录。
  size = vocab_size;
  
  // 重新计算词的个数。
  train_words = 0;
  
  // 遍历当前词典。
  for (a = 0; a < size; a++) {
    if ((vocab[a].cn < min_count) && (a != 0)) {
      // 减小新词典的长度。
      vocab_size--;
      
      // 释放词的内存。
      free(vocab[a].word);
    } else {
      // 重新计算 hash code。
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
   
  // 重新分配词典, 把表尾的所有低词频词都去掉。
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  
  // 给二叉树分配内存。
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

// 去掉低频次的词以减少词典的大小
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
    vocab[b].cn = vocab[a].cn;
    vocab[b].word = vocab[a].word;
    b++;
  } else free(vocab[a].word);
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // 重新计算 hash
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

/**
 * ======== CreateBinaryTree ========
 * Create binary Huffman tree using the word counts.
 * Frequent words will have short unique binary codes.
 * Huffman encoding is used for lossless compression.
 * For each vocabulary word, the vocab_word structure includes a `point` array, 
 * which is the list of internal tree nodes which:
 *   1. Define the path from the root to the leaf node for the word.
 *   2. Each correspond to a row of the output matrix.
 * The `code` array is a list of 0s and 1s which specifies whether each output
 * in `point` should be trained to output 0 or 1.
 */
void CreateBinaryTree() {
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH]; // Default is 40
  
  // Note that calloc initializes these arrays to 0.
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  
  // The count array is twice the size of the vocabulary, plus one.
  //   - The first half of `count` becomes a list of the word counts 
  //     for each word in the vocabulary. We do not modify this part of the
  //     list.
  //   - The second half of `count` is set to a large positive integer (1 
  //     quadrillion). When we combine two trees under a word (e.g., word_id 
  //     13), then we place the total weight of those subtrees into the word's
  //     position in the second half (e.g., count[vocab_size + 13]).
  //     
  for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
  
  // `pos1` and `pos2` are indeces into the `count` array.
  //   - `pos1` starts at the middle of `count` (the end of the list of word
  //     counts) and moves left.
  //   - `pos2` starts at the beginning of the list of large integers and moves
  //     right.
  pos1 = vocab_size - 1;
  pos2 = vocab_size;
  
  
  /* ===============================
   *   Step 1: Create Huffman Tree
   * ===============================
   * [Original Comment] Following algorithm constructs the Huffman tree by 
   * adding one node at a time
   * 
   * The Huffman coding algorithm starts with every node as its own tree, and
   * then combines the two smallest trees on each step. The weight of a tree is
   * the sum of the word counts for the words it contains. 
   * 
   * Once the tree is constructed, you can use the `parent_node` array to 
   * navigate it. For the word at index 13, for example, you would look at 
   * parent_node[13], and then parent_node[parent_node[13]], and so on, till
   * you reach the root.
   *
   * A Huffman tree stores all of the words in the vocabulary at the leaves.
   * Frequent words have short paths, and infrequent words have long paths.
   * Here, we are also associating each internal node of the tree with a 
   * row of the output matrix. Every time we combine two trees and create a 
   * new node, we give it a row in the output matrix.
   */  
  
  // The number of tree combinations needed is equal to the size of the vocab,
  // minus 1.
  for (a = 0; a < vocab_size - 1; a++) {

    // First, find two smallest nodes 'min1, min2'
    // Find min1 (at index `min1i`)
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }
    
    // Find min2 (at index `min2i`).
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    
    // Calculate the combined weight. We could be combining two words, a word
    // and a tree, or two trees.
    count[vocab_size + a] = count[min1i] + count[min2i];
    
    // Store the path for working back up the tree. 
    parent_node[min1i] = vocab_size + a;
    parent_node[min2i] = vocab_size + a;
    
    // binary[min1i] = 0; // This is implied.
    // min1 is the (left?) node and is labeled '0', min2 is the (right?) node
    // and is labeled '1'.
    binary[min2i] = 1;
  }

  /* ==========================================
   *    Step 2: Define Samples for Each Word
   * ==========================================
   * [Original Comment] Now assign binary code to each vocabulary word
   * 
   *  vocab[word]
   *    .code - A variable-length string of 0s and 1s.
   *    .point - A variable-length array of output row indeces.
   *    .codelen - The length of the `code` array. 
   *               The point array has length `codelen + 1`.
   * 
   */
    
  // For each word in the vocabulary...
  for (a = 0; a < vocab_size; a++) {
    b = a;
    i = 0; // `i` stores the code length.
    
    // Construct the binary code...
    //   `code` stores 1s and 0s.
    //   `point` stores indeces.
    // This loop works backwards from the leaf, so the `code` and `point` 
    // lists end up in reverse order.
    while (1) {
      // Lookup whether this is on the left or right of its parent node.
      code[i] = binary[b];
      
      // Note: point[0] always holds the word iteself...
      point[i] = b;
      
      // Increment the code length.
      i++;
      
      // This will always return an index in the second half of the array.
      b = parent_node[b];
      
      // We've reached the root when...
      if (b == vocab_size * 2 - 2) break;
    }
    
    // Record the code length (the length of the `point` list).
    vocab[a].codelen = i;
    
    // The root node is at row `vocab_size - 2` of the output matrix. 
    vocab[a].point[0] = vocab_size - 2;
    
    // For each bit in this word's code...
    for (b = 0; b < i; b++) {
      // Reverse the code in `code` and store it in `vocab[a].code`
      vocab[a].code[i - b - 1] = code[b];
      
      // Store the row indeces of the internal nodes leading to this word.
      // These are the set of outputs which will be trained every time
      // this word is encountered in the training data as an output word.
      vocab[a].point[i - b] = point[b] - vocab_size;      
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}

/**
 * ======== LearnVocabFromTrainFile ========
 * 从训练语料中建立词典,并建立散列表来快速查询所对应的词索引。
 *
 * 如果词频数量小于 min_reduce ，词会被丢弃掉。
 */
void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  
  // -1 填充散列表。
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  
  // 打开训练语料
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  
  vocab_size = 0;
  
  // 特殊标记 </s> 用于标记句子的末尾。
  // 在训练中，contexr window 不会超过句子的末尾。
  // 
  // 显式地添加 "</s>" 到词典中，所以让其在词典的0出出现。
  AddWordToVocab((char *)"</s>");
  
  while (1) {
    // 从文件读取一个词到 word 中。
    ReadWord(word, fin);
    
    // 当达到文件末尾时终止。
    if (feof(fin)) break;
    
    // 语料中词的个数。(词的个数，不是有多少不同的词)
    train_words++;
    
    // 每十万个词打印一次信息。
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    
    // 查看散列表是否已经添加了这个词。
    i = SearchVocab(word);
    
    // 不存在，添加该词。
    if (i == -1) {
      a = AddWordToVocab(word);
      
      // 初始化词频为1，此处有些迷惑。。直接Add的时候进行操作不行吗？
      vocab[a].cn = 1;
    
    // 如果已存在，增加词频。
    } else vocab[i].cn++;
    
    // 这个代码里的散列表是没有扩容功能的,所以如果词数量过多会频繁冲突。
    // 所以如果词的数量超过了散列表容量的70%，进行一次Reduce,即移除不频繁的词。
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }
  
  // 将词典按照词频降序排列
  // 移除词频小于 min_reduce的词。
  SortVocab();
  
  // 打印词典信息，词的个数(已经移除了不频繁的词)，有多少个不同的词
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  
  file_size = ftell(fin);
  fclose(fin);
}

void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

void ReadVocab() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    i++;
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}

/**
 * ======== InitNet ========
 *
 */
void InitNet() {
  long long a, b;
  unsigned long long next_random = 1;
  
  // 词向量层
  // syn0
  // syn0 = 词典的长度(vocab_size)*词向量的长度(layer1_size)
  // posix_memalign : 页对齐的内存。
  // posix_memalign() 成功时会返回size字节的动态内存，并且这块内存的地址是alignment(这里是128)的倍数
  a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
  
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
  
  // 如果用 hierarchical softmax 训练。
  if (hs) {
    a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++) 
        for (b = 0; b < layer1_size; b++)
            syn1[a * layer1_size + b] = 0;
  }
  
  // 如果用 negative sampling 训练。
  if (negative > 0) {
    // 分配网络的输出层。
    // 该层测变量为 syn1neg。
    // 这一层的大小与隐藏层相同，但是是转置的。
    // 如果以矩阵形式表达就是, 词向量的长度(layer1_size) * 词典的长度(vocab_size)。
    a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
    
    if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
    
    // 输出层的所有参数为0。
    for (a = 0; a < vocab_size; a++) 
        for (b = 0; b < layer1_size; b++)
            syn1neg[a * layer1_size + b] = 0;
  }
  
  // 随机初始化隐藏层的参数。
  // TODO - What's the equation here?
  // 词向量的参数为 [-0.5,0.5] 之间的小数。
  for (a = 0; a < vocab_size; a++) 
    for (b = 0; b < layer1_size; b++) {
        next_random = next_random * (unsigned long long)25214903917 + 11;
        syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
  }
  
  // 创建哈夫曼树,仅用于HS。
  CreateBinaryTree();
}

/**
 * ======== TrainModelThread ========
 * 这个函数在训练模型。
 */
void *TrainModelThread(void *id) {

  /*
   * word - 存储在词典中词的索引。
   * last_word - 上一个单词，辅助扫描窗口，就当前扫描到的上下文单词。
   * sentence_length - 当前处理的句子长度，当前句子的长度(词数)。
   * sentence_position - 当前处理的单词在当前句子中的位置(index)。
   * cw - 窗口长度。
   */
  long long a, b, d, cw, word, last_word, sentence_length = 0, sentence_position = 0;
  /*
   * word_count - 当前线程当前时刻已训练的词的个数。
   * last_word_count - 当前线程上一次已训练的词的个数。
   * sen - 单词数组，表示句子，从当前文件中读取的待处理的句子。
   */
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  /*
   * l1 -
   * l2 - 
   * c -
   * target -
   * label - 
   */
  long long l1, l2, c, target, label, local_iter = iter;
  // id 线程创建的时候传入，辅助随机数生成。
  unsigned long long next_random = (long long)id;
  real f, g;
  // 当前时间，和start比较计算算法效率。
  clock_t now;
  
  // neu1 只用于 CBOW 模型。
  real *neu1 = (real *)calloc(layer1_size, sizeof(real));
  
  // neu1e 两个模型都用。
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));
  
  
  // Open the training file and seek to the portion of the file that this 
  // thread is responsible for.
  FILE *fi = fopen(train_file, "rb");
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
  
  // 这个循环包括了整个训练流程。
  while (1) {
    
    /*
     * ======== Variables ========
     *       iter - 这是训练的 epoch 数，默认为5。
     * word_count - 要处理的词总数。
     * train_words - 训练文本中的单词总数(不包括
     *                 通过 ReduceVocab 从词典中删除的单词)。
     */
    
    // 这里打印训练信息，并且调整训练的"alpha" 参数。
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      
      last_word_count = word_count;
      
      // The percentage complete is based on the total number of passes we are
      // doing and not just the current pass.      
      if ((debug_mode > 1)) {
        now=clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
         // Percent complete = [# of input words processed] / 
         //                      ([# of passes] * [# of words in a pass])
         word_count_actual / (real)(iter * train_words + 1) * 100,
         word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      
      // Update alpha to: [initial alpha] * [percent of training remaining]
      // This means that alpha will gradually decrease as we progress through 
      // the training text.
      alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
      // Don't let alpha go below [initial alpha] * 0.0001.
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }
    
    // 这个判断语句检索下个句子并经其存储在"sen"中。
    if (sentence_length == 0) {
      while (1) {
        // Read the next word from the training data and lookup its index in 
        // the vocab table. 'word' is the word's vocab index.
        // 从训练数据中读取训练数据并在词典中查找其索引。
        // 'word' 是 word 词典索引。
        word = ReadWordIndex(fi);

        if (feof(fi)) break;
        
        // If the word doesn't exist in the vocabulary, skip it.
        if (word == -1) continue;
        
        // Track the total number of training words processed.
        word_count++;
        
        // 'vocab' word 0 is a special token "</s>" which indicates the end of 
        // a sentence.
        if (word == 0) break;
        
        /* 
         * =================================
         *   频繁词的下采样
         * =================================
         * 这段代码随机丢弃训练词，但是被设计成保持相对词频相同，也就是说，
         * 使用频率较低的词被丢弃的几率也较低。
         *
         * 我们首先要计算我们想要 "保留" 的词的概率 x。
         * 然后，为了决定是否保留这个单词，我们生成一个 (0,1) 的随机fraction，如果
         * 'ran' 小于这个数字，我们丢弃这个词。
         * 这意味着 'ran' 越小，我们越有可能丢弃这个单词。
         *
         * x = (vocab[word].cn / train_words) 一个词出现次数占所有词出现次数的比例。
         * 用 x 来表示这个fraction。
         *
         * sample的默认值是 0.001。
         * ran 的公式是：
         *   ran = (sqrt(x / 0.001) + 1) * (0.001 / x)
         * 
         * 你可以画出它的图来看一下，一个 L 曲线。
         * 
         * 下面是这个函数中一些有趣的点(同样，这里使用的是默认样本值 0.001)。
         *   - ran = 1 (100% 被留下) 即当 x<= 0.0026
         *      - 也就是说，任何 <= 0.0026 将肯定会被保留，
         *        仅对占词总数0.26%的词进行二次采样。
         *   - ran = 0.5 (50% 会被保留) when x = 0.00746. 
         *   - ran = 0.033 (3.3% 会被保留) when x = 1.
         *       - 也就是说，如果一个词占训练集的100%(当然，永远不会发生),
         *         那么该单词将仅保留 3.3%.
         *
         * NOTE: 所以看起来似乎计算每个单词的出现概率并将其存储在vocab表中会效率更高。
         *
         * 在下采样中被丢弃掉的词将不会添加到我们训练的句子中。
         * 这意味着这些被丢弃的词既不用做输入，也不用做其他输入的 context word.
         */
        if (sample > 0) {
          // 计算保留该词的概率
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          
          // 生成一个随机数
          // 64位整数
          next_random = next_random * (unsigned long long)25214903917 + 11;

          // 如果 随机数 > ran ，丢弃该词。
          //
          // (next_random & 0xFFFF) 抽取 next_random 的低16位随机数。
          // 将这个数除以 65536 (2^16) 给我们一个 (0,1) 的分数。
          // 所以这段代码只是生成一个随机分数。
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }
        
        // 如果我们保留词，则将其添加到句子里。
        sen[sentence_length] = word;
        sentence_length++;
        
        // 验证句子是否过长。
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      
      sentence_position = 0;
    }

    if (feof(fi) || (word_count > train_words / num_threads)) {
      word_count_actual += word_count - last_word_count;
      local_iter--;
      if (local_iter == 0) break;
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
      continue;
    }
    
    // 找出句子中的下一个单词。
    // 词由它在词汇表中的索引表示。
    word = sen[sentence_position];
    
    if (word == -1) continue;
    
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
    
    // 这是一个标准的随机整数生成器，如下所示：
    // https://en.wikipedia.org/wiki/Linear_congruential_generator
    next_random = next_random * (unsigned long long)25214903917 + 11;
    
    // b 是一个在 0 到 window -1 之间的随机整数。
    // 这是我们将缩小窗口大小的数量。
    b = next_random % window;
    
    /* 
     * ====================================
     *        CBOW Architecture
     * ====================================
     * sen - 这是由单词组表示的句子。已经经过了下采样。词由他们的id表示。
     *
     * sentence_position - This is the index of the current input word.
     *
     * a - Offset into the current window, relative to the window start.
     *     a will range from 0 to (window * 2) 
     *
     * b - The amount to shrink the context window by.
     *
     * c - 'c' is a scratch variable used in two unrelated ways:
     *       1. It's first used as the index of the current context word 
     *          within the sentence (the `sen` array).
     *       2. It's then used as the for-loop variable for calculating
     *          vector dot-products and other arithmetic.
     *
     * syn0 - The hidden layer weights. Note that the weights are stored as a
     *        1D array, so word 'i' is found at (i * layer1_size).
     *
     * target - The output word we're working on. If it's the positive sample
     *          then `label` is 1. `label` is 0 for negative samples.  
     *          Note: `target` and `label` are only used in negative sampling,
     *                and not HS.
     *
     * neu1 - This vector will hold the *average* of all of the context word
     *        vectors. This is the output of the hidden layer for CBOW.
     *
     * neu1e - Holds the gradient for updating the hidden layer weights.
     *         It's a vector of length 300, not a matrix.
     *         This same gradient update is applied to all context word 
     *         vectors.
     */
    if (cbow) {  //train the cbow architecture
      // in -> hidden
      cw = 0;
      
      // This loop will sum together the word vectors for all of the context
      // words.
      //
      // Loop over the positions in the context window (skipping the word at
      // the center). 'a' is just the offset within the window, it's not the 
      // index relative to the beginning of the sentence.
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {

        // Convert the window offset 'a' into an index 'c' into the sentence 
        // array.
        c = sentence_position - window + a;
        
        // Verify c isn't outisde the bounds of the sentence.
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        
        // Get the context word. That is, get the id of the word (its index in
        // the vocab table).
        last_word = sen[c];
        
        // At this point we have two words identified:
        //   'word' - The word (word ID) at our current position in the 
        //            sentence (in the center of a context window).
        //   'last_word' - The word (word ID) at a position within the context
        //                 window.       
        
        // Verify that the word exists in the vocab
        if (last_word == -1) continue;
        
        // Add the word vector for this context word to the running sum in 
        // neur1.
        // `layer1_size` is 300, `neu1` is length 300
        for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + last_word * layer1_size];
        
        // Count the number of context words.
        cw++;
      }
      
      // Skip if there were somehow no context words.
      if (cw) {
        
        // neu1 was the sum of the context word vectors, and now becomes
        // their average. 
        for (c = 0; c < layer1_size; c++) neu1[c] /= cw;
        
        // // HIERARCHICAL SOFTMAX
        // vocab[word]
        //   .point - A variable-length list of row ids, which are the output
        //            rows to train on.
        //   .code - A variable-length list of 0s and 1s, which are the desired
        //           labels for the outputs in `point`.
        //   .codelen - The length of the `code` array for this 
        //              word.
        // 
        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
          f = 0;
          // point[d] is the index of a row of the ouput matrix.
          // l2 is the index of that word in the output layer weights (syn1).
          l2 = vocab[word].point[d] * layer1_size;
          
          // Propagate hidden -> output
          // neu1 is the average of the context words from the hidden layer.
          // This loop computes the dot product between neu1 and the output
          // weights for the output word at point[d].
          for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1[c + l2];
          
          // Apply the sigmoid activation to the current output neuron.
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          
          // 'g' is the error multiplied by the learning rate.
          // The error is (label - f), so label = (1 - code), meaning if
          // code is 0, then this is a positive sample and vice versa.
          g = (1 - vocab[word].code[d] - f) * alpha;
          // Propagate errors output -> hidden
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
          // Learn weights hidden -> output
          for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c];
        }
        
        // NEGATIVE SAMPLING
        // Rather than performing backpropagation for every word in our 
        // vocabulary, we only perform it for the positive sample and a few
        // negative samples (the number of words is given by 'negative').
        // These negative words are selected using a "unigram" distribution, 
        // which is generated in the function InitUnigramTable.        
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          // On the first iteration, we're going to train the positive sample.
          if (d == 0) {
            target = word;
            label = 1;
          
          // On the other iterations, we'll train the negative samples.
          } else {
            // Pick a random word to use as a 'negative sample'; do this using 
            // the unigram table.
            
            // Get a random integer.              
            next_random = next_random * (unsigned long long)25214903917 + 11;
            
            // 'target' becomes the index of the word in the vocab to use as
            // the negative sample.            
            target = table[(next_random >> 16) % table_size];
            
            // If the target is the special end of sentence token, then just
            // pick a random word from the vocabulary instead.            
            if (target == 0) target = next_random % (vocab_size - 1) + 1;

            // Don't use the positive sample as a negative sample!            
            if (target == word) continue;
            
            // Mark this as a negative example.
            label = 0;
          }
          
          // At this point, target might either be the positive sample or a 
          // negative sample, depending on the value of `label`.
          
          // Get the index of the target word in the output layer.
          l2 = target * layer1_size;
          
          // Calculate the dot product between:
          //   neu1 - The average of the context word vectors.
          //   syn1neg[l2] - The output weights for the target word.
          f = 0;
          for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];

          // This block does two things:
          //   1. Calculates the output of the network for this training
          //      pair, using the expTable to evaluate the output layer
          //      activation function.
          //   2. Calculate the error at the output, stored in 'g', by
          //      subtracting the network output from the desired output, 
          //      and finally multiply this by the learning rate.          
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          
          // Multiply the error by the output layer weights.
          // (I think this is the gradient calculation?)
          // Accumulate these gradients over all of the negative samples.          
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];

          // Update the output layer weights by multiplying the output error
          // by the average of the context word vectors.
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];
        }
         
        // hidden -> in
        // Backpropagate the error to the hidden layer (the word vectors).
        // This code is used both for heirarchical softmax and for negative
        // sampling.
        //
        // Loop over the positions in the context window (skipping the word at
        // the center). 'a' is just the offset within the window, it's not 
        // the index relative to the beginning of the sentence.
        for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
          // Convert the window offset 'a' into an index 'c' into the sentence 
          // array.
          c = sentence_position - window + a;
          
          // Verify c isn't outisde the bounds of the sentence.
          if (c < 0) continue;
          if (c >= sentence_length) continue;
          
          // Get the context word. That is, get the id of the word (its index in
          // the vocab table).
          last_word = sen[c];
          
          // Verify word exists in vocab.
          if (last_word == -1) continue;
          
          // Note that `c` is no longer the sentence position, it's just a 
          // for-loop index.
          // Add the gradient in the vector `neu1e` to the word vector for
          // the current context word.
          // syn0[last_word * layer1_size] <-- Accesses the word vector.
          for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c];
        }
      }
    } 
    /* 
     * ====================================
     *        Skip-gram Architecture
     * ====================================
     * sen - 由词组成的句子，数组表示。已经做了高频词的下采样。
     *       词由他们的索引表示。
     *
     * sentence_position - 当前输入词在句子中的位置。
     *
     * a - 当前窗口的下标，相对于window的起始索引。
     *     a 的范围从[0 , window * 2]。
     *
     * b - 要缩小的 context window 大小。
     *
     * c - 'c' 是一个 scratch 变量 以两种不相关的方式使用。
     *     1. 它首先用作当前context(w) 在句子中的索引(sen 数组)
     *     2. 然后，它被用作计算向量点积和其他算数运算的for循环变量。 
     *
     * syn0 - 隐藏层的参数。注意，权重存储为一个一维数组，
     *        因此单词 'i' 在 (i * layer1_size) 找到。
     *
     * l1 - 索引到隐藏层(syn0)。当前输入词的权重开头的索引。
     *
     * target - 我们正在训练的输出词。
     *          如果它是正样本，那么 “标签” 是1。
     *          注: “target” 和 “label” 只用于 Negative sampling，而非HS。
     */
    else {  
      /*
       * 遍历 context window 中的位置 (跳过中心词)。
       * a 是在 context window 内的相对索引，不是句子的开始索引。
       */
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        // 将 context window 内的相对索引 'a' 转换成句子中的相对索引 'c'。
        c = sentence_position - window + a;
        
        // 确定c 的索引在句子的范围内。
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        
        // 获取 context word ，也就是说，获取单词的id(在词典中的索引)。
        last_word = sen[c];
        
        // 至此，我们已经确定了两个词
        //   'word' - 在句子中当前位置的单词。(context window 的中心)
        // 
        //   'last_word' - 位于 context window 内某个位置的词。
        
        // 验证词是否存在于词典中。

        // 在这一点上，我们确定了两个词:
        // 'word' - 在句子中当前位置的词(在上下文窗口的中心位置)。
        // 'last_word' - 位于 context window 内某个位置的单词。
        if (last_word == -1) continue;
        
        // 计算 'last_word' 的索引。
        l1 = last_word * layer1_size;
        
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
        
        // HIERARCHICAL SOFTMAX
        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
          f = 0;
          l2 = vocab[word].point[d] * layer1_size;
          // Propagate hidden -> output
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1[c + l2];
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          // 'g' is the gradient multiplied by the learning rate
          g = (1 - vocab[word].code[d] - f) * alpha;
          // Propagate errors output -> hidden
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
          // Learn weights hidden -> output
          for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * syn0[c + l1];
        }
        
        // NEGATIVE SAMPLING
        // 我们不会对词典中的每个词都执行反向传播，而是只对少数词执行反向传播
        // (单词的数量由'negative'给出)。
        // 这些单词是使用'unigram' 分布进行选择的，
        // 该分布式在函数 InitUnigramTable 中生成的。
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          // 第一次迭代我们去训练正样本。
          if (d == 0) {
            target = word;
            label = 1;
          // 其他的迭代训练负样本。
          } else {
            // 在能量表中随机抽取负样本，采样使得与target不同，label为0,
            // 也即最多采样 negative 个负样本。
            // 获取一个随机整数。
            next_random = next_random * (unsigned long long)25214903917 + 11;
            
            // 'target' 成为词汇中单词的索引，作为负样本使用。
            target = table[(next_random >> 16) % table_size];
            
            // 如果目标是特殊的句子结束标记，那么只需从词汇表中随机选择一个词即可。
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            
            // 不要把正样本做负样本使用!
            if (target == word) continue;
            
            // 标记为负样本
            label = 0;
          }
          
          // 获取 target word 在输出层的索引。
          l2 = target * layer1_size;
          
          // At this point, our two words are represented by their index into
          // the layer weights.
          // l1 - The index of our input word within the hidden layer weights.
          // l2 - The index of our output word within the output layer weights.
          // label - Whether this is a positive (1) or negative (0) example.
          
          // Calculate the dot-product between the input words weights (in 
          // syn0) and the output word's weights (in syn1neg).
          // Note that this calculates the dot-product manually using a for
          // loop over the vector elements!
          f = 0;
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
          
          // This block does two things:
          //   1. Calculates the output of the network for this training
          //  g rate.
          if (f > MAX_EXP) g = (label - 1) * alpha;    pair, using the expTable to evaluate the output layer
          //      activation function.
          //   2. Calculate the error at the output, stored in 'g', by
          //      subtracting the network output from the desired output, 
          //      and finally multiply this by the learning rate.
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          
          // Multiply the error by the output layer weights.
          // Accumulate these gradients over the negative samples and the one
          // positive sample.
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
          
          // Update the output layer weights by multiplying the output error
          // by the hidden layer weights.
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
        }
        // Once the hidden layer gradients for the negative samples plus the 
        // one positive sample have been accumulated, update the hidden layer
        // weights. 
        // Note that we do not average the gradient before applying it.
        for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
      }
    }
    
    // Advance to the next word in the sentence.
    sentence_position++;
    
    // Check if we've reached the end of the sentence.
    // If so, set sentence_length to 0 and we'll read a new sentence at the
    // beginning of this loop.
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  free(neu1);
  free(neu1e);
  pthread_exit(NULL);
}

/**
 * ======== TrainModel ========
 * 训练流程的主入口。
 */
void TrainModel() {
  long a, b, c, d;
  FILE *fo;
  
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  
  printf("Starting training using file %s\n", train_file);
  
  starting_alpha = alpha;
  
  // 要么加载已有的词典，要么训练词典。
  if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
  
  // 保存词典。
  if (save_vocab_file[0] != 0) SaveVocab();
  
  // 如果没有指定输出文件从这里退出。
  if (output_file[0] == 0) return;
  
  // 初始化权重矩阵。
  InitNet();

  // 如果我们使用 negative sampling , 初始化单词能量表，
  // 用于 negative samples的采样。
  if (negative > 0) InitUnigramTable();
  
  // 记录训练开始时间。
  start = clock();
  
  // 开始训练，即 TrainModelThread 函数
  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
  
  
  fo = fopen(output_file, "wb");
  if (classes == 0) {
    // Save the word vectors
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    for (a = 0; a < vocab_size; a++) {
      fprintf(fo, "%s ", vocab[a].word);
      if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
      else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
      fprintf(fo, "\n");
    }
  } else {
    // Run K-means on the word vectors
    int clcn = classes, iter = 10, closeid;
    int *centcn = (int *)malloc(classes * sizeof(int));
    int *cl = (int *)calloc(vocab_size, sizeof(int));
    real closev, x;
    real *cent = (real *)calloc(classes * layer1_size, sizeof(real));
    for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;
    for (a = 0; a < iter; a++) {
      for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
      for (b = 0; b < clcn; b++) centcn[b] = 1;
      for (c = 0; c < vocab_size; c++) {
        for (d = 0; d < layer1_size; d++) cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
        centcn[cl[c]]++;
      }
      for (b = 0; b < clcn; b++) {
        closev = 0;
        for (c = 0; c < layer1_size; c++) {
          cent[layer1_size * b + c] /= centcn[b];
          closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
        }
        closev = sqrt(closev);
        for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
      }
      for (c = 0; c < vocab_size; c++) {
        closev = -10;
        closeid = 0;
        for (d = 0; d < clcn; d++) {
          x = 0;
          for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
          if (x > closev) {
            closev = x;
            closeid = d;
          }
        }
        cl[c] = closeid;
      }
    }
    // Save the K-means classes
    for (a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);
    free(centcn);
    free(cent);
    free(cl);
  }
  fclose(fo);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
    printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
    printf("\t-hs <int>\n");
    printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 12)\n");
    printf("\t-iter <int>\n");
    printf("\t\tRun more training iterations (default 5)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
    printf("\t-classes <int>\n");
    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\t-cbow <int>\n");
    printf("\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
    printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
    return 0;
  }
  output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
  if (cbow) alpha = 0.05;
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
  
  // 分配词典
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  
  // Allocate the hash table for mapping word strings to word entries.
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
   
  /*
   * ======== Precomputed Exp Table ========
   * To calculate the softmax output, they use a table of values which are
   * pre-computed here.
   *
   * From the top of this file:
   *   #define EXP_TABLE_SIZE 1000
   *   #define MAX_EXP 6
   *
   * First, let's look at this inner term:
   *     i / (real)EXP_TABLE_SIZE * 2 - 1
   * This is just a straight line that goes from -1 to +1.
   *    (0, -1.0), (1, -0.998), (2, -0.996), ... (999, 0.998), (1000, 1.0).
   *
   * Next, multiplying this by MAX_EXP = 6, it causes the output to range
   * from -6 to +6 instead of -1 to +1.
   *    (0, -6.0), (1, -5.988), (2, -5.976), ... (999, 5.988), (1000, 6.0).
   *
   * So the total input range of the table is 
   *    Range = MAX_EXP * 2 = 12
   * And the increment on the inputs is
   *    Increment = Range / EXP_TABLE_SIZE = 0.012
   *
   * Let's say we want to compute the output for the value x = 0.25. How do
   * we calculate the position in the table?
   *    index = (x - -MAX_EXP) / increment
   * Which we can re-write as:
   *    index = (x + MAX_EXP) / (range / EXP_TABLE_SIZE)
   *          = (x + MAX_EXP) / ((2 * MAX_EXP) / EXP_TABLE_SIZE)
   *          = (x + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2)
   *
   * The last form is what we find in the code elsewhere for using the table:
   *    expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
   * 
   */

  // Allocate the table, 1000 floats.
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  
  // For each position in the table...
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    
    // Calculate the output of e^x for values in the range -6.0 to +6.0.
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    
    // Currently the table contains the function exp(x).
    // We are going to replace this with exp(x) / (exp(x) + 1), which is
    // just the sigmoid activation function! 
    // Note that 
    //    exp(x) / (exp(x) + 1) 
    // is equivalent to 
    //    1 / (1 + exp(-x))
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  
  TrainModel();
  return 0;
}
