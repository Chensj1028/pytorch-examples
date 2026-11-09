from time import time # Track how long an epoch takes
# 记录一轮训练所用时间。
import os # Creating and finding files/directories
# 创建和查找文件与目录。
import logging # Logging tools
# 日志工具。
from datetime import date # Logging the date for model versioning
# 记录日期，用于模型版本管理。

import torch # For ML
# 机器学习相关工具。
from tqdm import tqdm # For fancy progress bars
# 用于显示进度条。

from src.model import Translator # Our model
# 本例模型。
from src.data import get_data, create_mask, generate_square_subsequent_mask # Loading data and data preprocessing
# 加载数据并进行预处理。
from argparse import ArgumentParser # For args
# 命令行参数处理。

# Train on the GPU if possible
# 尽可能使用 GPU 训练。
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to generate output sequence using greedy algorithm
# 使用贪心算法生成输出序列的函数。
def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol):

    # Move to device
    # 移动到计算设备。
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    # Encode input
    # 编码输入。
    memory = model.encode(src, src_mask)

    # Output will be stored here
    # 输出保存在这里。
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)

    # For each element in our translation (which could range up to the maximum translation length)
    # 逐个生成译文元素，最多生成到指定的最大翻译长度。
    for _ in range(max_len-1):

        # Decode the encoded representation of the input
        # 解码已编码的输入表示。
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0), DEVICE).type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)

        # Reshape
        # 调整形状。
        out = out.transpose(0, 1)

        # Covert to probabilities and take the max of these probabilities
        # 转换为概率，并选择概率最大的项。
        prob = model.ff(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        # Now we have an output which is the vector representation of the translation
        # 现在得到的输出是译文的向量表示。
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == end_symbol:
            break

    return ys

# Opens an user interface where users can translate an arbitrary sentence
# 打开交互界面，允许用户翻译任意句子。
def inference(opts):

    # Get training data, tokenizer and vocab
    # 获取训练数据、分词器和词表，
    # objects as well as any special symbols we added to our dataset
    # 以及添加到数据集中的特殊符号。
    _, _, src_vocab, tgt_vocab, src_transform, _, special_symbols = get_data(opts)

    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    # Create model
    # 创建模型。
    model = Translator(
        num_encoder_layers=opts.enc_layers,
        num_decoder_layers=opts.dec_layers,
        embed_size=opts.embed_size,
        num_heads=opts.attn_heads,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        dim_feedforward=opts.dim_feedforward,
        dropout=opts.dropout
    ).to(DEVICE)

    # Load in weights
    # 加载权重。
    model.load_state_dict(torch.load(opts.model_path))

    # Set to inference
    # 设置为推理模式。
    model.eval()

    # Accept input and keep translating until they quit
    # 持续接收输入并翻译，直到用户退出。
    while True:
        print("> ", end="")

        sentence = input()

        # Convert to tokens
        # 转换为词元。
        src = src_transform(sentence).view(-1, 1)
        num_tokens = src.shape[0]

        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)

        # Decode
        # 解码。
        tgt_tokens = greedy_decode(
            model, src, src_mask, max_len=num_tokens+5, start_symbol=special_symbols["<bos>"], end_symbol=special_symbols["<eos>"]
        ).flatten()

        # Convert to list of tokens
        # 转换为词元列表。
        output_as_list = list(tgt_tokens.cpu().numpy())

        # Convert tokens to words
        # 将词元转换为单词。
        output_list_words = tgt_vocab.lookup_tokens(output_as_list)

        # Remove special tokens and convert to string
        # 移除特殊词元，并转换为字符串。
        translation = " ".join(output_list_words).replace("<bos>", "").replace("<eos>", "")

        print(translation)

# Train the model for 1 epoch
# 训练模型一轮。
def train(model, train_dl, loss_fn, optim, special_symbols, opts):

    # Object for accumulating losses
    # 用于累积损失的对象。
    losses = 0

    # Put model into training mode
    # 将模型设为训练模式。
    model.train()
    for src, tgt in tqdm(train_dl, ascii=True):

        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        # We need to reshape the input slightly to fit into the transformer
        # 稍微调整输入形状，以符合 Transformer 的要求。
        tgt_input = tgt[:-1, :]

        # Create masks
        # 创建掩码。
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, special_symbols["<pad>"], DEVICE)

        # Pass into model, get probability over the vocab out
        # 送入模型，得到词表上各项的输出分数。
        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        # Reset gradients before we try to compute the gradients over the loss
        # 计算损失梯度之前先清空梯度。
        optim.zero_grad()

        # Get original shape back
        # 恢复原来的形状。
        tgt_out = tgt[1:, :]

        # Compute loss and gradient over that loss
        # 计算损失及其梯度。
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        # Step weights
        # 更新权重。
        optim.step()

        # Accumulate a running loss for reporting
        # 累积损失，用于输出进度。
        losses += loss.item()

        if opts.dry_run:
            break

    # Return the average loss
    # 返回平均损失。
    return losses / len(list(train_dl))

# Check the model accuracy on the validation dataset
# 在验证数据集上检查模型准确率。
def validate(model, valid_dl, loss_fn, special_symbols):
    
    # Object for accumulating losses
    # 用于累积损失的对象。
    losses = 0

    # Turn off gradients a moment
    # 暂时关闭梯度跟踪。
    model.eval()

    for src, tgt in tqdm(valid_dl):

        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        # We need to reshape the input slightly to fit into the transformer
        # 稍微调整输入形状，以符合 Transformer 的要求。
        tgt_input = tgt[:-1, :]

        # Create masks
        # 创建掩码。
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, special_symbols["<pad>"], DEVICE)

        # Pass into model, get probability over the vocab out
        # 送入模型，得到词表上各项的输出分数。
        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        # Get original shape back, compute loss, accumulate that loss
        # 恢复原来的形状，计算并累积损失。
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    # Return the average loss
    # 返回平均损失。
    return losses / len(list(valid_dl))

# Train the model
# 训练模型。
def main(opts):

    # Set up logging
    # 配置日志。
    os.makedirs(opts.logging_dir, exist_ok=True)
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=opts.logging_dir + "log.txt", level=logging.INFO)

    # This prints it to the screen as well
    # 同时将日志打印到屏幕。
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    logging.info(f"Translation task: {opts.src} -> {opts.tgt}")
    logging.info(f"Using device: {DEVICE}")

    # Get training data, tokenizer and vocab
    # 获取训练数据、分词器和词表，
    # objects as well as any special symbols we added to our dataset
    # 以及添加到数据集中的特殊符号。
    train_dl, valid_dl, src_vocab, tgt_vocab, _, _, special_symbols = get_data(opts)

    logging.info("Loaded data")

    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    logging.info(f"{opts.src} vocab size: {src_vocab_size}")
    logging.info(f"{opts.tgt} vocab size: {tgt_vocab_size}")

    # Create model
    # 创建模型。
    model = Translator(
        num_encoder_layers=opts.enc_layers,
        num_decoder_layers=opts.dec_layers,
        embed_size=opts.embed_size,
        num_heads=opts.attn_heads,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        dim_feedforward=opts.dim_feedforward,
        dropout=opts.dropout
    ).to(DEVICE)

    logging.info("Model created... starting training!")

    # Set up our learning tools
    # 设置训练所需工具。
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=special_symbols["<pad>"])

    # These special values are from the "Attention is all you need" paper
    # 这些特殊数值来自 Attention Is All You Need 论文。
    optim = torch.optim.Adam(model.parameters(), lr=opts.lr, betas=(0.9, 0.98), eps=1e-9)

    best_val_loss = 1e6
    
    for idx, epoch in enumerate(range(1, opts.epochs+1)):

        start_time = time()
        train_loss = train(model, train_dl, loss_fn, optim, special_symbols, opts)
        epoch_time = time() - start_time
        val_loss   = validate(model, valid_dl, loss_fn, special_symbols)

        # Once training is done, we want to save out the model
        # 训练结束后保存模型。
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logging.info("New best model, saving...")
            torch.save(model.state_dict(), opts.logging_dir + "best.pt")

        torch.save(model.state_dict(), opts.logging_dir + "last.pt")

        logger.info(f"Epoch: {epoch}\n\tTrain loss: {train_loss:.3f}\n\tVal loss: {val_loss:.3f}\n\tEpoch time = {epoch_time:.1f} seconds\n\tETA = {epoch_time*(opts.epochs-idx-1):.1f} seconds")

if __name__ == "__main__":

    parser = ArgumentParser(
        prog="Machine Translator training and inference",
    )

    # Inference mode
    # 推理模式。
    parser.add_argument("--inference", action="store_true",
                        help="Set true to run inference")
    parser.add_argument("--model_path", type=str,
                        help="Path to the model to run inference on")

    # Translation settings
    # 翻译设置。
    parser.add_argument("--src", type=str, default="de",
                        help="Source language (translating FROM this language)")
    parser.add_argument("--tgt", type=str, default="en",
                        help="Target language (translating TO this language)")

    # Training settings 
    # 训练设置。
    parser.add_argument("-e", "--epochs", type=int, default=30,
                        help="Epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Default learning rate")
    parser.add_argument("--batch", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--backend", type=str, default="cpu",
                        help="Batch size")
    
    # Transformer settings
    # Transformer 设置。
    parser.add_argument("--attn_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--enc_layers", type=int, default=5,
                        help="Number of encoder layers")
    parser.add_argument("--dec_layers", type=int, default=5,
                        help="Number of decoder layers")
    parser.add_argument("--embed_size", type=int, default=512,
                        help="Size of the language embedding")
    parser.add_argument("--dim_feedforward", type=int, default=512,
                        help="Feedforward dimensionality")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Transformer dropout")

    # Logging settings
    # 日志设置。
    parser.add_argument("--logging_dir", type=str, default="./" + str(date.today()) + "/",
                        help="Where the output of this program should be placed")

    # Just for continuous integration
    # 仅用于持续集成。
    parser.add_argument("--dry_run", action="store_true")

    args = parser.parse_args()

    DEVICE = torch.device("cuda" if args.backend == "gpu" and torch.cuda.is_available() else "cpu")

    if args.inference:
        inference(args)
    else:
        main(args)
