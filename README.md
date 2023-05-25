BỘ GIÁO DỤC VÀ ĐÀO TẠO
TRƯỜNG ĐẠI HỌC QUỐC TẾ SÀI GÒN
KHOA: KHOA HỌC MÁY TÍNH & KỸ THUẬT


TIỂU LUẬN CUỐI KỲ

NĂM HỌC 2022 – 2023
TÊN ĐỀ TÀI: ỨNG DỤNG NLP TRONG VIỆC ĐÁNH GIÁ BÌNH LUẬN CỦA NGƯỜI DÙNG

Giảng viên: Phan Thị Thể


  Thành viên:




Thành phố Hồ Chí Minh
Mục Lục:
CHƯƠNG 1: GIỚI THIỆU	4
1. Bối cảnh nghiên cứu.	4
2. NLP là gì?	4
3. Sentiment Analysis là gì?	5
4. Ứng dụng Sentiment Analysis vào việc đánh giá bình luận người dùng.	5
CHƯƠNG 2: HƯỚNG TIẾP CẬN, TÓM TẮT KHÁI QUÁT VỀ MÔ HÌNH VÀ CÁCH HÌNH THÀNH MÔ HÌNH	7
I. HƯỚNG TIẾP CẬN	7
II. THÔNG TIN VỀ MÔ HÌNH ĐƯỢC SỬ DỤNG	8
III. CÁCH HÌNH THÀNH MÔ HÌNH	9
1. Thu thập dữ liệu:	9
2. Tiền xử lý dữ liệu:	13
3. Code bài toán	22
CHƯƠNG 3. THỰC TIỄN ÁP DỤNG CÔNG NGHỆ PHÂN TÍCH TÌNH CẢM TRONG VIỆC ĐÁNH GIÁ BÌNH LUẬN CỦA NGƯỜI DÙNG	26
CHƯƠNG 4. KẾT LUẬN VÀ TRIỂN VỌNG	27
1. Kết Luận	27
2. Triển vọng	29














Hình 1. Giới thiệu về NLP	5
Hình 2. Flow chart của thu thập dữ liệu	11
Hình 3. Bảng sau khi được loại bỏ các cột không cần thiết	14
Hình 4. Biểu đồ cột thể hiện số lượng data của mỗi bình luận	16
Hình 5. Hình ảnh của bảng sau khi được loại bỏ các nhận xét giống nhau	17
Hình 6. Biểu đồ cột sau khi loại bỏ các giá trị bị trùng	18


 
CHƯƠNG 1: GIỚI THIỆU
1. Bối cảnh nghiên cứu.
Trong thời đại hiện nay công nghệ thông tin phát triển mạnh và đã đóng vai trò không thể thiếu trong cuộc sống hiện nay. Ảnh hưởng của việc này đã dẫn đến các phương thức bán hàng online, hoặc các dịch vụ online ra đời, ví dụ như: các buổi livestream bán quần áo, thức ăn nhanh,..vv., hoặc dịch vụ đặt trước phòng của các khách sạn. 
Từ những việc trên đã phát sinh ra một bài toán đó là các bình luận đánh giá của người dùng sao cho những phần được khách hàng đánh giá sẽ thu hút nhiều khách hàng hơn. Vì vậy các chủ doanh nghiệp, cửa hàng phải quan tâm nhiều hơn đến các bình luận đánh giá đó để cải thiện chất lượng sản phẩm dịch vụ tốt hơn cho khách hàng. Nhưng con người không thể nào sát sao đánh giá được từ vài trăm bình luận trở nên, vì thế cho nên việc đánh giá các bình luận này trở nên khó khăn.
Để xử lý vấn đề này, NLP - xử lý ngôn ngữ tự nhiên là một công cụ hữu ích trong việc đánh giá các bình luận một các hiệu quả nhất. Việc sử dụng NLP giúp doanh nghiệp có thể nhanh chóng phân tích các ý kiến của khách hàng về sản phẩm hoặc dịch vụ của họ, từ các đánh giá bình luận đó có thể tạo ra các phản hồi và giúp cải thiện lại chất lượng sản phẩm, dịch vụ khách hàng.
Trong bài báo cáo này chúng em sử dụng một hướng phát triển của NLP là Sentiment Analysis.
2. NLP là gì? 
Natural Language Processing (NLP) hay còn gọi là xử lý ngôn ngữ tự nhiên. Thuật ngữ này đề cập đến một nhánh của khoa học máy tính, cụ thể hơn là nhánh của trí tuệ nhân tạo (AI). Nó cung cấp cho máy tính khả năng hiểu văn bản và lời nói của con người.
NLP điều khiển các chương trình máy tính dịch văn bản từ ngôn ngữ này sang ngôn ngữ khác, phản hồi các lệnh được nói và tóm tắt khối lượng lớn văn bản một cách nhanh chóng ngay cả trong thời gian thực. 
 
Hình 1. Giới thiệu về NLP
Chúng ta thường tương tác với NLP dưới dạng hệ thống GPS điều khiển bằng giọng nói, trợ lý kỹ thuật số, phần mềm đọc chính tả, chuyển giọng nói thành văn bản, chatbots dịch vụ khách hàng và các tiện ích tiêu dùng khác. Nhưng NLP cũng đóng một vai trò ngày càng lớn trong các giải pháp doanh nghiệp giúp hợp lý hóa hoạt động kinh doanh, tăng năng suất của nhân viên và đơn giản hóa các quy trình kinh doanh quan trọng.
3. Sentiment Analysis là gì?
Sentiment Analysis – Phân tích cảm xúc là  một kỹ thuật sử dụng xử lý ngôn ngữ tự nhiên và học máy để xác định và định lượng cảm xúc và ý kiến được bày tỏ trong một đoạn văn bản. Nó có thể giúp bạn hiểu cách mọi người cảm thấy về một chủ đề, sản phẩm hoặc dịch vụ bằng cách phân tích phản hồi, đánh giá hoặc nhận xét của họ. Phân tích cảm xúc có thể được thực hiện ở các cấp độ chi tiết khác nhau, chẳng hạn như tài liệu, câu hoặc khía cạnh. Nó cũng có thể được thực hiện bằng các phương pháp khác nhau, chẳng hạn như dựa trên quy tắc, tự động hoặc hệ thống lai.
4. Ứng dụng Sentiment Analysis vào việc đánh giá bình luận người dùng.
Công nghệ Sentiment Analysis có thể được áp dụng vào việc đánh giá phân tích tình cảm của người dùng trong các bình luận, đánh giá, phản hồi về sản phẩm hoặc dịch vụ của doanh nghiệp. Cụ thể, các thuật toán Sentiment Analysis có thể được sử dụng để phân tích ngôn ngữ tự nhiên của người dùng, xác định những từ hoặc cụm từ mang tính tích cực, tiêu cực hoặc trung lập, từ đó đưa ra nhận xét hoặc đánh giá về sản phẩm hoặc dịch vụ đó.
Các ứng dụng của công nghệ Sentiment Analysis trong việc phân tích tình cảm bao gồm:
•	Tự động phân loại bình luận hoặc đánh giá thành tích cực, tiêu cực hoặc trung lập để đánh giá chất lượng sản phẩm hoặc dịch vụ của doanh nghiệp.
•	Xác định những từ khóa hoặc cụm từ mang tính cảm xúc và phân tích tần suất xuất hiện của chúng trong các bình luận của người dùng để đưa ra đánh giá chính xác hơn về sự hài lòng của khách hàng với sản phẩm hoặc dịch vụ.
•	Tự động phát hiện và loại bỏ các bình luận tiêu cực hoặc không phù hợp với sản phẩm hoặc dịch vụ của doanh nghiệp.
Tuy nhiên, việc sử dụng công nghệ Sentiment Analysis trong việc đánh giá phân tích tình cảm cũng có một số hạn chế như:
•	Khả năng đánh giá phân tích tình cảm không hoàn toàn chính xác do các ngôn ngữ đa nghĩa và sự khác biệt về ngôn ngữ sử dụng giữa các vùng miền, quốc gia.
•	Sự chính xác của kết quả phân tích tình cảm còn phụ thuộc vào chất lượng của dữ liệu đầu vào và thuật toán sử dụng.
Tuy nhiên, với sự phát triển của công nghệ Sentiment Analysis, việc áp dụng nó trong việc đánh giá phân tích tình cảm ngày càng được sử dụng rộng rãi và cải thiện đáng kể chất lượng của phân tích tình cảm.



 
CHƯƠNG 2: HƯỚNG TIẾP CẬN, TÓM TẮT KHÁI QUÁT VỀ MÔ HÌNH VÀ CÁCH HÌNH THÀNH MÔ HÌNH
I. HƯỚNG TIẾP CẬN
a. Lexicon-based
Phương pháp dựa trên từ điển các từ thể hiện cảm xúc. Theo đó, việc dự đoán cảm xúc dựa vào việc tìm kiếm các từ cảm xúc riêng lẻ, xác định điểm số cho các từ tích cực, xác định điểm số cho các từ tiêu cực và sau đó là tổng hợp các điểm số này lại theo một độ đo xác định để quyết định xem văn bản màu sắc cảm xúc gì. Phương pháp này có điểm hạn chế là thứ tự các từ bị bỏ qua và các thông tin quan trọng có thể bị mất. Độ chính xác của mô hình phụ thuộc vào độ tốt của bộ từ điển các từ cảm xúc. Nhưng lại có ưu điểm là dễ thực hiện, chi phí tính toán nhanh, chỉ mất công sức trong việc xây dựng bộ từ điển các từ cảm xúc mà thôi.
b. Deep Learning
Những thập niên gần đây, với sự phát triển nhanh chóng tốc độ xử lý của CPU, GPU và chi phí cho phần cứng ngày càng giảm, các dịch vụ hạ tầng điện toán đám mây ngày càng phát triển, làm tiền đề và cơ hội cho phương pháp học sâu Deep Learning Neural Network phát triển mạnh mẽ. Trong đó, bài toán phân tích cảm xúc đã được giải quyết bằng mô hình học Recurrent Neural Network (RNN) với một biến thể được dùng phổ biến hiện nay là Long Short Term Memory Neural Network (LSTMs), kết hợp với mô hình vector hóa từ (vector representations of words) Word2Vector với kiến trúc Continuous Bag-of-Words (CBOW). Mô hình này cho độ chính xác hơn 85%. Ưu điểm của phương pháp này là văn bản đầu vào có thể là 1 câu hay 1 đoạn văn. Để thực hiện mô hình này đòi hỏi phải có dữ liệu văn bản càng nhiều càng tốt để tạo Word2Vector CBOW chất lượng cao và dữ liệu gán nhãn lớn để huấn luyện (training), xác minh (validate) và kiểm tra (test) mô hình học có giám sát (Supervise Learning) LSTMs. Một phương pháp khác là sử dụng Transformer với nhiều pre-train model khác nhau và đây cũng là phương pháp được sử dụng trong bài báo cáo này với model được sử dụng là DistilBERT.
c. Phương pháp kết hợp Rule-bases (dựa trên luật) và Corpus-bases (dựa trên ngữ liệu). 
Tiêu biểu cho phương pháp này là công trình nghiên cứu của Richard Socher thuộc trường đại học Stanford. Phương pháp này kết hợp sử dụng mô hình Deep Learning Recursive Neural Network với hệ tri thức chuyên gia trong xử lý ngôn ngữ tự nhiên (XLNNTN) được gọi là Sentiment Treebank. Sentiment Tree là cây phân tích cú pháp của 1 câu văn, trong đó mỗi nút trong cây kèm theo bộ trọng số cảm xúc lần lượt là: rất tiêu cực (very negative), tiêu cực (negative), trung tính (neutral), tích cực (positive) và rất tích cực (very positive). Theo đó, trọng số thuộc nhãn nào lớn nhất sẽ quyết định nhãn toàn cục của nút, như hình dưới đây. Độ chính xác của mô hình khi dự đoán cảm xúc cho 1 câu đơn là 85,4%. Nhược điểm của phương pháp này ở chổ chỉ xử lý tốt cho dữ liệu đầu vào là một câu đơn.
II. THÔNG TIN VỀ MÔ HÌNH ĐƯỢC SỬ DỤNG
1. Tìm hiểu về BERT
BERT là viết tắt của cụm từ Bidirectional Encoder Representation from Transformer có nghĩa là mô hình biểu diễn từ theo 2 chiều ứng dụng kỹ thuật Transformer. BERT được thiết kế để huấn luyện trước các biểu diễn từ (pre-train word embedding). Điểm đặc biệt ở BERT đó là nó có thể điều hòa cân bằng bối cảnh theo cả 2 chiều trái và phải.
Cơ chế attention của Transformer sẽ truyền toàn bộ các từ trong câu văn đồng thời vào mô hình một lúc mà không cần quan tâm đến chiều của câu. Do đó Transformer được xem như là huấn luyện hai chiều (bidirectional) mặc dù trên thực tế chính xác hơn chúng ta có thể nói rằng đó là huấn luyện không chiều (non-directional). Đặc điểm này cho phép mô hình học được bối cảnh của từ dựa trên toàn bộ các từ xung quanh nó bao gồm cả từ bên trái và từ bên phải.

	CoLA 	MNLI	MRPC	QNLI	QQP	RTE	SST-2	STS-B	WNLI
BERT	56.3	86.7	88.6	91.8	89.6	69.3	92.7	89.0	53.5
Bảng 1. Điểm số của BERT
III. CÁCH HÌNH THÀNH MÔ HÌNH
1. Thu thập dữ liệu:
a. Dữ liệu mẫu:
Bộ dữ liệu chứa nhận xét từ người dùng và các thông tin chung của sản phẩm từ Amazon, chứa 142,8 triệu dữ liệu trong khoảng thời gian tháng 5 năm 1996 đến tháng 6 năm 2014.
Bộ dữ liệu được sử dụng gồm 20GB dưới dạng file JSON bao gồm các mục:
•	reviewerID – Mã của người nhận xét
•	asin  – Mã sản phẩm được nhận xét
•	reviewerName – Tên người nhận xét
•	helpful – Độ hữu dụng
•	reviewText  – Nhận xét (chữ)
•	overall – Đánh giá (sao)
•	summary – Tóm tắt
•	unixReviewTime – Thời gian Unix của nhận xét
•	reviewTime – Thời gian quốc tế của nhận xét
b. Dữ liệu thực tế:
Đây sẽ là bộ dữ liệu được thu thập từ các trang web bán hàng như Amazon, Ebay. Các dữ liệu được thu tập thông qua Beautiful Soup, 1 thư viện của Python, có khả năng lấy dữ liệu từ các trang HTML/XML, và xuất dưới dạng file csv. Cách dữ liệu được xử lý thể hiện qua work flow bên dưới:
 
Hình 2. Flow chart của thu thập dữ liệu
Code thu thập dữ liệu:
Thêm các thư viện cần thiết

import requests
from bs4 import BeautifulSoup
import pandas as pd

Tạo biến header để đưa vào l

HEADERS = ({'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) \AppleWebKit/537.36 (KHTML, like Gecko) \Chrome/90.0.4430.212 Safari/537.36','Accept-Language': 'en-US, en;q=0.5'})

Tạo biến url để đưa vào lệnh

url = 'https://www.amazon.sg/John-Constantine-Hellblazer-Vol-Family/product-reviews/1401236901/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews'

Tạo hàm để lấy file html của trang web

def getdata(url):
	r = requests.get(url, headers=HEADERS)
	return r.text

Tạo hàm để parse text

def html_code(url):
	htmldata = getdata(url)
	soup = BeautifulSoup(htmldata, 'html.parser')
	return (soup)

tạo hàm để lấy review

def cus_review(soup):
    data_str = ""
    for item in soup.find_all("div", attrs = ("class","a-row a-spacing-small review-data")):
        data_str = data_str + item.get_text()
    result = data_str.split("\n")
    return (result)
  

Tạo và in file của webpage

soup = html_code(url)
print(soup)

tạo list các review và skip các review không có chữ

rev_data = cus_review(soup)
rev_result = []
for i in rev_data:
	if i in '':
		pass
	else:
		rev_result.append(i)


In các list ra

print(rev_result[6])

len(rev_result)

Đưa list về table và xuất file csv

# initialise data of lists.
data = {'review': rev_result}

# Create DataFrame
df = pd.DataFrame(data)

# Save the output.
df.to_csv('amazon_review.csv')
2. Tiền xử lý dữ liệu:
Với data tương đối lớn việc huấn luyện sẽ yêu cầu việc pre-process là cực kì quan trọng. Đầu tiên ta làm sạch data bằng việc lọc các cột không cần thiết và loại bỏ chúng thông qua hàm drop và chúng ta sẽ bỏ các cột 'asin', 'reviewerName', 'helpful', 'summary', 'unixReviewTime', 'reviewTime', 'reviewerID', df = df.drop(['asin', 'reviewerName', 'helpful', 'summary', 'unixReviewTime', 'reviewTime', 'reviewerID'], axis=1) đây sẽ là các cột bị loại bỏ vì không cần thiết. Chúng ta sẽ còn lại cột review định dạng chữ và cột rating định dạng số. Sử dụng df.head() ta sẽ kiểm tra được 5 phần tử đầu
 
Hình 3. Bảng sau khi được loại bỏ các cột không cần thiết
Tiếp tục chúng ta kiểm tra số thành phần trong mỗi rating bằng lệnh count:
overall_count=df.overall.value_counts()

Các giá trị Rating	Số lượng phần tử thuộc mỗi giá trị
1.0
	360821
2.0	255666

3.0
	429054

4.0
	950534
5.0
	3003925

Bảng 3. Thống kê số lượng các giá trị
Để dễ dàng hình dung dữ liệu, chúng ta sẽ vẽ 1 biểu đồ cột để thể hiện chúng:
plt.bar(overall_count.index,overall_count.values)
plt.xlabel('Rating')
plt.ylabel('Value')
plt.show()
Đây là dòng lệnh để vẽ nên 1 biểu đồ cột với trục tung là số lượng phần tử thuộc mỗi giá trị và trục hoành là các giá trị rating và code này sẽ cho ta biểu đồ dưới đây:
 
Hình 4. Biểu đồ cột thể hiện số lượng data của mỗi bình luận
Thông qua hình mẫu này ta có thể thấy được nhiều người phản ứng tính cực với các sản phẩm trong dataset nhưng từ 3 dòng đầu ta có thể thấy rằng có sự giống nha của data vậy nên ta phải xoá những data giống nhau thông qua hàm drop_duplicates
df. drop_duplicates()
 
Hình 5. Hình ảnh của bảng sau khi được loại bỏ các nhận xét giống nhau
Từ đây ta có thể tính toán và vẽ lại 1 bảng và biểu đồ hoàn chỉnh hơn
Các giá trị Rating	Số lượng phần tử thuộc mỗi giá trị
1.0
	235029
2.0	149242
3.0
	249393
4.0
	549571
5.0
	1733211
Bảng 4.Thống kê số lượng các giá trị sao khi loại bỏ các giá trị trùng
 
Hình 6. Biểu đồ cột sau khi loại bỏ các giá trị bị trùng
Các giá trị bị trùng chiếm đến khoảng ½ số lượng giá trị chính xác từ đó có thể dẫn tới sai sót trong việc phân tích dữ liệu.
Để kiểm tra số lượng các từ vựng nào được sử dụng nhiều nhất, ta có thể sử dụng Word Cloud để có thể tìm các từ vựng đặc biệt
 
Hình 7. Word Cloud
Sau khi remove các rating và thay thế bằng label ta sẽ vẽ được biểu đồ sau
 
Hình 8. Data theo label
	Tiếp tục khảo sát data ta có thể thấy rằng sẽ ra sao nếu chúng ta vẽ một biểu đồ thể hiện số lượng từ nào có nhiều review nhất
 
Hình 9. Chiều dài của từ và số lượng
 
3. Code bài toán
from datasets import load_dataset
imdb = load_dataset("imdb")
imdb["test"][0]
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)
tokenized_imdb = imdb.map(preprocess_function, batched=True)
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
import evaluate

accuracy = evaluate.load("accuracy")
import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}
from transformers import create_optimizer
import tensorflow as tf

batch_size = 20
num_epochs = 15
batches_per_epoch = len(tokenized_imdb["train"]) // batch_size
total_train_steps = int(batches_per_epoch * num_epochs)
optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)
from transformers import TFAutoModelForSequenceClassification

model = TFAutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
)
tf_train_set = model.prepare_tf_dataset(
    tokenized_imdb["train"],
    shuffle=True,
    batch_size=16,
    collate_fn=data_collator,
)

tf_validation_set = model.prepare_tf_dataset(
    tokenized_imdb["test"],
    shuffle=False,
    batch_size=16,
    collate_fn=data_collator,
)
import tensorflow as tf

model.compile(optimizer=optimizer)
from transformers.keras_callbacks import KerasMetricCallback

metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_validation_set)
from transformers.keras_callbacks import PushToHubCallback

push_to_hub_callback = PushToHubCallback(
    output_dir="my_awesome_model",
    tokenizer=tokenizer,
)
text = "90% of this was the situational emotion, and I didn't connect. It became a formula of several fighting encounters, too many new characters and the High Evolutionary villain screaming every line until there was no more scenery to chew. Yo Space Adrian!"
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")
classifier(text)
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_model")

inputs = tokenizer(text, return_tensors="tf")
from transformers import TFAutoModelForSequenceClassification

model = TFAutoModelForSequenceClassification.from_pretrained("stevhliu/my_awesome_model")

logits = model(**inputs).logits
predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])

model.config.id2label[predicted_class_id]
CHƯƠNG 3. THỰC TIỄN ÁP DỤNG CÔNG NGHỆ PHÂN TÍCH TÌNH CẢM TRONG VIỆC ĐÁNH GIÁ BÌNH LUẬN CỦA NGƯỜI DÙNG
Chuẩn bị dữ liệu: Thu thập và tiền xử lý dữ liệu train của bạn. Điều này có thể bao gồm việc làm sạch, phân mảnh thành các từ riêng biệt (tokenizing) và định dạng dữ liệu văn bản để phù hợp với yêu cầu đầu vào của mô hình Transformer. Mô hình distilbert-base-uncased sẽ được sử dụng thông qua thư viện Transformer. 
Tokenization: Phân mảnh dữ liệu văn bản đã được tiền xử lý thành biểu diễn số hóa mà có thể được đưa vào mô hình Transformer. Thư viện Transformer cung cấp các tiện ích tokenization mà bạn có thể sử dụng cho mục đích này. Chúng ta sẽ sử dụng AutoTokenizer
Đánh giá: Đánh giá hiệu suất của mô hình đã được huấn luyện bằng cách đánh giá trên một tập dữ liệu kiểm tra hoặc xác nhận riêng biệt. Điều này giúp bạn hiểu được mô hình hoạt động tốt như thế nào trên dữ liệu chưa được nhìn thấy và cho phép bạn điều chỉnh siêu tham số nếu cần thiết.
Chạy thực nghiệm: Sau khi mô hình của bạn đã được huấn luyện và đánh giá, bạn có thể sử dụng nó để chạy dự đoán. Thư viện Transformer cung cấp phương pháp pipeline để thực hiện Inference bằng cách sử dụng mô hình đã được huấn luyện.
Trong quá trình tiền xử lý dữ liệu và huấn luyện mô hình Transformer, có một số lưu ý và thách thức cần quan tâm:
Dữ liệu huấn luyện: Đảm bảo rằng dữ liệu huấn luyện của bạn đủ lớn và đa dạng để mô hình có thể học được các khía cạnh và ngữ cảnh phong phú. Nếu dữ liệu huấn luyện quá nhỏ, mô hình có thể gặp khó khăn trong việc tổng quát hóa cho dữ liệu mới.
Tiền xử lý dữ liệu: Quá trình tiền xử lý dữ liệu có thể phức tạp và tốn thời gian. Bạn cần xử lý các vấn đề như làm sạch dữ liệu, xử lý ngôn ngữ tự nhiên (NLP) như chia câu, phân mảnh thành từ (tokenization) và loại bỏ stop words. Đồng thời, cũng cần xử lý 
Tối ưu hóa mô hình: Mô hình distilbert-base-uncased có nhiều siêu tham số (hyperparameters) cần được tinh chỉnh để đạt hiệu suất tốt nhất. Bạn có thể điều chỉnh số lớp, số head attention, kích thước lớp ẩn và tốc độ học (learning rate) để cải thiện khả năng tổng quát hóa và độ chính xác của mô hình

CHƯƠNG 4. KẾT LUẬN VÀ TRIỂN VỌNG
1. Kết Luận
Công nghệ Nhận diện từ khóa đang trở thành công cụ hữu ích trong việc đánh giá bình luận của người dùng trên các nền tảng mạng xã hội. Bằng cách sử dụng các thuật toán NLP, công nghệ này có thể nhận diện các từ khóa quan trọng trong các bình luận và đánh giá của người dùng để giúp doanh nghiệp có thể phân tích và đánh giá các vấn đề liên quan đến sản phẩm hoặc dịch vụ của mình.

Việc sử dụng công nghệ Nhận diện từ khóa trong đánh giá bình luận của người dùng mang lại nhiều lợi ích cho doanh nghiệp, bao gồm:
a.	Hiểu được những đánh giá của khách hàng và phản hồi kịp thời trước các vấn đề phát sinh.
b.	Định hướng các chiến lược tiếp thị và phát triển sản phẩm, dịch vụ để đáp ứng nhu cầu của khách hàng.
c.	Nâng cao chất lượng phục vụ, tăng cường lòng tin của khách hàng đối với thương hiệu của doanh nghiệp.
Tuy nhiên, công nghệ Nhận diện từ khóa cũng có những hạn chế, chẳng hạn như khả năng hiểu sai ý đồ của người dùng khi sử dụng các từ ngữ không rõ ràng hoặc khi sử dụng các ngôn ngữ khác nhau. Để khắc phục những hạn chế này, cần phải kết hợp công nghệ Nhận diện từ khóa với các công nghệ khác để đạt hiệu quả cao nhất.

Phân tích tình cảm là một công nghệ NLP quan trọng trong việc đánh giá các bình luận của người dùng. Điều này giúp cho các doanh nghiệp có thể đánh giá được cảm xúc của người dùng đối với sản phẩm hoặc dịch vụ của mình. Công nghệ này sử dụng các thuật toán và mô hình học máy để phân tích và đánh giá tình cảm của các từ và câu trong bình luận.

Các lợi ích của việc sử dụng công nghệ phân tích tình cảm bao gồm:

•	Đánh giá được cảm xúc của người dùng: giúp các doanh nghiệp hiểu được người dùng đánh giá sản phẩm hoặc dịch vụ của mình như thế nào.
•	Cải thiện chất lượng sản phẩm và dịch vụ: từ việc đánh giá được tình cảm của người dùng, các doanh nghiệp có thể cải thiện chất lượng sản phẩm và dịch vụ để đáp ứng nhu cầu của khách hàng.
•	Tiết kiệm thời gian và chi phí: công nghệ phân tích tình cảm giúp tự động hóa quá trình đánh giá bình luận, tiết kiệm thời gian và chi phí so với việc thực hiện bằng tay.	

Mặc dù công nghệ phân tích tình cảm có rất nhiều lợi ích, nhưng nó cũng tồn tại một số hạn chế:

•	Độ chính xác của phân tích tình cảm phụ thuộc rất nhiều vào độ chính xác của dữ liệu đầu vào và thuật toán phân tích. Việc thu thập và xử lý dữ liệu có thể ảnh hưởng đến kết quả phân tích tình cảm.

•	Sự khác biệt về ngôn ngữ và văn hóa là một thách thức lớn đối với phân tích tình cảm. Các thuật toán phân tích tình cảm cần phải được đào tạo trên nhiều ngôn ngữ và đa dạng văn hóa để có thể đưa ra kết quả chính xác.

•	Phân tích tình cảm không thể thay thế hoàn toàn cho con người trong việc đánh giá các bình luận của người dùng. Các phương pháp phân tích tình cảm chỉ có thể đưa ra các kết quả chung chung và không thể hiểu được mối quan hệ giữa các từ và câu trong bình luận.

Tuy nhiên, với sự tiến bộ của công nghệ và phương pháp học máy, chúng ta có thể mong đợi rằng phân tích tình cảm sẽ được phát triển và cải thiện trong tương lai, giúp cho doanh nghiệp có thể đánh giá bình luận của người dùng một cách chính xác và đáng tin cậy hơn.
2. Triển vọng
Trong tương lai, công nghệ này sẽ thêm các hướng phát triển như:
Nhận diện từ khóa: NLP cũng có thể nhận diện các từ khóa quan trọng trong các bình luận và đánh giá của người dùng để giúp doanh nghiệp có thể phân tích và đánh giá các vấn đề liên quan đến sản phẩm hoặc dịch vụ của mình. 
Phân tích ý kiến: NLP cũng có thể phân tích ý kiến của người dùng trong các bình luận, đánh giá để giúp doanh nghiệp hiểu rõ hơn về ý kiến và suy nghĩ của khách hàng về sản phẩm hoặc dịch vụ của họ.
Tóm tắt văn bản: NLP có thể tóm tắt các bình luận, đánh giá của người dùng để giúp cho doanh nghiệp có thể nhanh chóng hiểu và đưa ra phản hồi phù hợp.
Dịch thuật: NLP cũng có thể dịch các bình luận, đánh giá từ một ngôn ngữ sang ngôn ngữ khác để giúp cho doanh nghiệp có thể đánh giá được ý kiến của khách hàng trên toàn thế giới.





 
TÀI LIỆU THAM KHẢO
https://machinelearningcoban.com/tabml_book/ch_intro/pipeline.html
























