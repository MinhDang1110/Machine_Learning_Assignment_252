def load_ag_news(path):
    """
    Đọc dữ liệu AG News, xử lý header và ghép title + description thành cột text.

    Input:
        path: thư mục chứa train.csv và test.csv

    Output:
        train_df, test_df
    """
    train_path = os.path.join(path, "train.csv")
    test_path = os.path.join(path, "test.csv")

    train_df = pd.read_csv(
        train_path,
        header=0,
        names=["label", "title", "description"]
    )

    test_df = pd.read_csv(
        test_path,
        header=0,
        names=["label", "title", "description"]
    )

    for df in [train_df, test_df]:
        # Đảm bảo label là kiểu số
        df["label"] = pd.to_numeric(df["label"], errors="coerce")

        # Loại bỏ dòng lỗi label
        df.dropna(subset=["label"], inplace=True)

        # Chuyển nhãn từ [1, 4] về [0, 3]
        df["label"] = df["label"].astype(int) - 1

        # Ghép title + description thành text
        df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")

        # Reset index sau khi drop
        df.reset_index(drop=True, inplace=True)

    return train_df, test_df


def plot_eda(df, label_names):
    """
    Vẽ biểu đồ phân phối nhãn và phân phối độ dài văn bản.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Phân phối nhãn
    sns.countplot(x=df["label"], ax=axes[0])
    axes[0].set_xticks(range(len(label_names)))
    axes[0].set_xticklabels(label_names, rotation=0)
    axes[0].set_title("Class Distribution")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Count")

    # Phân phối độ dài văn bản
    text_len = df["text"].apply(lambda x: len(str(x).split()))
    sns.histplot(text_len, bins=50, ax=axes[1], kde=True)
    axes[1].set_title("Text Length Distribution")
    axes[1].set_xlabel("Number of words")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()


def plot_wordcloud(df, label_idx, label_name, sample_size=2000):
    """
    Vẽ WordCloud cho một lớp cụ thể.
    """
    class_texts = df[df["label"] == label_idx]["text"]

    if len(class_texts) == 0:
        print(f"No text found for label {label_idx}: {label_name}")
        return

    sample_n = min(sample_size, len(class_texts))
    text = " ".join(class_texts.sample(sample_n, random_state=42).astype(str))

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white"
    ).generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title(f"WordCloud for class: {label_name}")
    plt.axis("off")
    plt.show()
