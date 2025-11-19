# Neko-LoRA-Build-Your-AI-Catgirl
这是一个Lora微调LLM模型的简单范例，供AI大模型初学者在有Nvidia显卡的WindowsPC上学习

如果看不懂Lora/LLM/Nvidia或者其他用到的缩写是什么，请查询https://chat.deepseek.com/ 求教一下鲸鱼老师


我们从头说起：

1、你需要一张英伟达的显卡
首先你的PC需要有一张还说得过去的Nvidia的独立显卡（起码得有个3G或者4G的显存吧）并安装了比较新版本的驱动，没有或者用AMD的……爱莫能助

2、在你的电脑上安装CUDA。
建议在安装CUDA前先更新显卡驱动到新版，这样可以直接安装高版本的CUDA。高版本CUDA兼容低版本，装个高的总不会错。

PS：在cmd或者powershell中输入：
nvidia-smi 
这个看的是物理机的显卡驱动版本和最兼容的cuda版本（A）
nvcc -V
这个看的是当前安装的cuda版本（B）。
按理来说应该满足A大于B的；实际上A和B在一个大版本就行，我们经常A=12.2，但装个B=12.8，或者A=11.4，但B=11.8，只要代码不报错就是可以的。

3、创建虚拟环境
建议用conda做一下环境隔离，避免同个机器上多个环境之间互相干扰。对于此项目，推荐使用Python=3.12
如何新建一个制定python版本的环境/激活/在指定路径创建请咨询鲸鱼老师
虚拟环境是好文明。如果在公用服务器上把base环境搞崩了，请自行做好挨揍的准备。

4、安装这个项目需要的环境
在激活你为这个项目准备的虚拟环境后：
如果你安装的CUDA版本是12.8，那么在win11下你可以直接点击运行env.bat
bat env.bat
（也许你需要一些权限）
如果你的cuda版本不是12.8或者你是在Linux机器上，大部分情况下：
pip3 install -r requirements.txt
是可以的。
如果不行的话你需要按照requirements.txt中由上到下的顺序，依次安装这些库。这需要你根据你的CUDA版本调整
PS：这个项目并没有用到什么高版本的特性，我认为只要满足Transformers能识别Qwen3这个模型就行了。
PPS：在pip安装的时候，如果有一些关于缺少某库的依赖/版本不对的报错，但你的code能运行，那不管也行
PPPS：一般显示“not an error with pip”的错误，都可以通过预安装需要的依赖/直接安装预编译版本解决。但GPU驱动版本太低这种硬伤是pip上无论如何耍杂技也解决不了的。所以先打好驱动装好新CUDA再来配环境

5、使用训练好的模型聊天
如果你的独立显卡有4GB以上的显存的话（我不清楚是否可以自动CPUoffload）——
在装好依赖的环境中运行
python3 infer_cat.py
你就可以和我训练出来的猫娘对话了
emmm，她可能不大聪明。
毕竟base模型只是个Qwen3 1.7B，而我们的鲸鱼老师是671B，四百倍的差距呢。

6、训练你自己的猫娘
如果你的独立显卡有至少6GB的显存（8GB是肯定可以的），那么你可以运行
python3 train_CAT.py
亲自体验Qwen3-1.7B被调教成猫娘的过程x
你可以自由的调整train_CAT.py中的那些参数，观察训练时各种指标的变化。

7、其他文件的说明
load_datasets_CAT.py 用于检查数据集的格式
merge.py Lora微调不会改变原始权重，会额外保存一些权重。你可以把Lora微调理解成“原始权重”+“我的权重”。所以在infer中，这两部分全中国是分开加载的。这个文件可以把这两部分权重融合到一起。

8、参考文献
数据集来源：
猫娘数据集https://huggingface.co/datasets/liumindmind/NekoQA-10K/tree/main
猫娘数据集论文https://zhuanlan.zhihu.com/p/1934983798233231689?share_code=1ed1vtHkWmGWV&utm_psn=1967708606477607700
（发表于国际顶刊：zhihu）

（下面这数据集留给你自己探索，你可以试着模仿本项目训练猫娘的方式，训练一个自己的弱智x）
（提示：需要检查数据集内容，然后设置一下outputmessage和inputmessage）
Yuki弱智吧问答数据集https://www.modelscope.cn/datasets/DanKe123abc/yuki_ruozhiba_1.5k/file/view/master/yuki_ruozhiba_1.5k.jsonl?id=145745&status=1

Qwen3-1.7B的开源权重地址
https://huggingface.co/Qwen/Qwen3-1.7B
如果你告诉我打不开这个link的话。。。那么：
https://modelscope.cn/models/Qwen/Qwen3-1.7B-Base

如果你的显存比较紧张，1.7B模型都训练不了。那么你可以考虑下Qwen3-0.6B。显然，参数又少了2/3，模型肯定是更加智障了。
https://huggingface.co/Qwen/Qwen3-0.6B
https://modelscope.cn/models/Qwen/Qwen3-0.6B-Base
用这个的话你需要修改一下infer与train中的模型路径



PS：估算微调大模型需要的显存
假设你要微调的原始模型为xB
对于Lora微调，约需要3x~4x的显存
对于QLora微调，至少也需要2.5x的显存
对于全量微调，采用AdamW优化器，大约需要20x以上的显存
如果你只需要推理，理论上全量参数（一般为BF16格式）需要2.5x显存，8bit量化需要1.2x，4bit量化需要0.6x
（以上针对仅语言模型，对多模态模型，显存需要会更多）

举例：
微调一个7B语言模型：
Lora微调，24GB不保证行，32GB应该可以；
QLora微调，16GB也许能星星；
全量微调，H20-141G可能不够，需要A100-80GBx2；
仅推理，BF16格式至少需要16G，8bit至少8G，4bit至少4G