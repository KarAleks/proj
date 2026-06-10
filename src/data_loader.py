import pandas as pd
import matplotlib.pyplot as plt

# This part is generated with AI
#####################################
columns = [
    # basic connection features (1-9)
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent",
    # content features (10-22)
    "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    # time-based traffic features (23-31)
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    # host-based traffic features (32-41)
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    # labels
    "label", "difficulty",
]

DOS_ATTACKS = {
    "back", "land", "neptune", "pod", "smurf", "teardrop",
    "apache2", "udpstorm", "processtable", "worm",
}
PROBE_ATTACKS = {
    "ipsweep", "nmap", "portsweep", "satan", "mscan", "saint",
}
R2L_ATTACKS = {
    "ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy",
    "warezclient", "warezmaster", "sendmail", "named", "snmpgetattack",
    "snmpguess", "xlock", "xsnoop", "httptunnel",
}
U2R_ATTACKS = {
    "buffer_overflow", "loadmodule", "perl", "rootkit",
    "ps", "sqlattack", "xterm",
}

####################################

train_data_path = "./data/NSL-KDD/KDDTrain+.txt"
test_data_path = "./data/NSL-KDD/KDDTest+.txt"

def map_attack_category(label):
    label = label.strip().lower()
    if label == "normal":
        return "normal"
    elif label in DOS_ATTACKS:
        return "DoS"
    elif label in PROBE_ATTACKS:
        return "Probe"
    elif label in R2L_ATTACKS:
        return "R2L"
    elif label in U2R_ATTACKS:
        return "U2R"
    return "DoS"

def load_datasets(train_path, test_path):
    train_data = pd.read_csv(train_path, header=None,names=columns)
    test_data = pd.read_csv(test_path, header=None, names=columns)
    
    train_data["binary_label"] = train_data["label"] != "normal"
    test_data["binary_label"] = test_data["label"] != "normal"
    
    train_data["attack_cat"] = train_data["label"].map(map_attack_category)
    test_data["attack_cat"] = test_data["label"].map(map_attack_category)
    
    print("Train data: ", train_data.shape)
    print("Test data: ", test_data.shape)

    # print(train_data.isnull().sum())
    # print(test_data.isnull().sum())
    return train_data, test_data
    
def stats(df, phase="Train"):
    fig, (ax1, ax2) = plt.subplots(1,2 ,figsize=(13, 5))
    
    order = ["normal", "DoS", "Probe", "R2L", "U2R"]
    counts = df["attack_cat"].value_counts().reindex(order)
    # print(counts)
    bars = ax1.bar(counts.index, counts.values)
    ax1.set_title("Attack Category Distribution")
    ax1.set_xlabel("Category")
    ax1.set_ylabel("Count")
        
    binary_counts = df["binary_label"].value_counts()
    binary_counts = binary_counts.rename({0: "Normal", 1: "Attack"})
    ax2.pie(binary_counts.values, labels=binary_counts.index, autopct="%1.1f%%", colors=["steelblue", "tomato"], startangle=90)
    ax2.set_title("Normal vs Attack")

    plt.tight_layout()
    plt.savefig(f"results/class_distribution_{phase}.png", dpi=150, bbox_inches="tight")

train_data, test_data = load_datasets(train_data_path, test_data_path)
stats(train_data, phase="Train")
stats(test_data, phase="Test")
     
