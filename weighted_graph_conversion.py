import argparse
import gzip
import torch

from transformers import AutoModel

SAVE_MODE = 0


def load_model(model_name):
    model = AutoModel.from_pretrained(model_name)
    model.eval()  # inference mode

    model_info = model.config.to_dict()

    return model, model_info


def add_token_to_preattn_edges(edges, E):

    V_dim = E.shape[0]
    H_dim = E.shape[1]

    for i in range(V_dim):
        for j in range(H_dim):

            src = f"Tok:{i}"
            target = f"H0:{j}"
            weight = float(E[i, j])

            edges.append((src, target, weight))


def add_position_to_preattn_edges(edges, E):

    P_dim = E.shape[0]
    H_dim = E.shape[1]

    for i in range(P_dim):
        for j in range(H_dim):

            src = f"Pos:{i}"
            target = f"H0:{j}"
            weight = float(E[i, j])

            edges.append((src, target, weight))


def add_type_to_preattn_edges(edges, E):

    T_dim = E.shape[0]
    H_dim = E.shape[1]

    for i in range(T_dim):
        for j in range(H_dim):

            src = f"Typ:{i}"
            target = f"H0:{j}"
            weight = float(E[i, j])

            edges.append((src, target, weight))


def add_layernorm_gamma_self_edges(
    edges,
    gamma,
    node_prefix,
):
    g = gamma.detach().cpu().view(-1)
    H = g.numel()
    for i in range(H):
        edges.append((f"{node_prefix}:{i}", f"{node_prefix}:{i}", float(g[i])))


def add_H_to_QKV_connections(
    edges,
    W_out_in,  # nn.Linear.weight (shape [out, in]), transpose required...
    num_heads: int,
    src_prefix: str,
    block_prefix: str,
    Wtype: str,  # "Q" | "K" | "V"
):

    W = W_out_in.t().contiguous()
    H = W.shape[0]
    d_k = H // num_heads

    # separate heads
    W_heads = torch.split(W, d_k, dim=1)

    for h, W_h in enumerate(W_heads):
        for i in range(H):
            for j in range(d_k):
                edges.append(
                    (
                        f"{src_prefix}:{i}",
                        f"{block_prefix}:HEAD{h}:{Wtype}:{j}",
                        float(W_h[i, j]),
                    )
                )


def add_headsout_to_postattn_edges(
    edges,
    block_layer,
    block_tag: str,
):
    WO = block_layer.attention.output.dense.weight.t().contiguous()
    H_in, H_out = WO.shape

    for i in range(H_in):
        for j in range(H_out):
            edges.append(
                (
                    f"{block_tag}:HeadsOut:{i}",
                    f"{block_tag}:PostAttn:{j}",
                    float(WO[i, j]),
                )
            )


def add_postattn_to_ffn_edges(
    edges,
    block_layer,
    block_tag: str,
):
    W1 = block_layer.intermediate.dense.weight.t().contiguous()
    H_in, d_ff = W1.shape
    for i in range(H_in):
        for j in range(d_ff):
            edges.append(
                (f"{block_tag}:PostAttn:{i}", f"{block_tag}:FFN:{j}", float(W1[i, j]))
            )


def add_ffn_to_postffn_edges(
    edges,
    block_layer,
    block_tag: str,
):
    W2 = block_layer.output.dense.weight.t().contiguous()
    d_ff, H_out = W2.shape
    for i in range(d_ff):
        for j in range(H_out):
            edges.append(
                (f"{block_tag}:FFN:{i}", f"{block_tag}:PostFFN:{j}", float(W2[i, j]))
            )


def add_postattn_to_pooler_edges(edges, m, src_prefix: str, target_prefix: str):
    WP = m.pooler.dense.weight.t().contiguous()
    H_in, H_out = WP.shape

    for i in range(H_in):
        for j in range(H_out):
            edges.append((f"{src_prefix}:{i}", f"{target_prefix}:{j}", float(WP[i, j])))


def add_block_edges(edges, model, block_layer_idx, src_prefix):

    layer = model.encoder.layer[block_layer_idx]
    block_tag = f"B{block_layer_idx}"
    num_heads = model.config.num_attention_heads

    before = len(edges)

    # ATTENTION STAGE: H(k)_src -> Q/K/V (chain terminates here)
    add_H_to_QKV_connections(
        edges, layer.attention.self.query.weight, num_heads, src_prefix, block_tag, "Q"
    )
    add_H_to_QKV_connections(
        edges, layer.attention.self.key.weight, num_heads, src_prefix, block_tag, "K"
    )
    add_H_to_QKV_connections(
        edges, layer.attention.self.value.weight, num_heads, src_prefix, block_tag, "V"
    )

    # INTERMEDIATE STAGE: HeadsOut -> PostAttn via W_O; then LN gamma
    add_headsout_to_postattn_edges(edges, layer, block_tag)
    add_layernorm_gamma_self_edges(
        edges, layer.attention.output.LayerNorm.weight, f"{block_tag}:PostAttn"
    )

    # PostAttn -> FFN via W1
    add_postattn_to_ffn_edges(edges, layer, block_tag)

    # OUTPUT STAGE: FFN -> PostFFN via W2; then LN gamma
    add_ffn_to_postffn_edges(edges, layer, block_tag)
    add_layernorm_gamma_self_edges(
        edges, layer.output.LayerNorm.weight, f"{block_tag}:PostFFN"
    )

    after = len(edges)
    added = after - before

    # Next block’s source is this block’s PostFFN
    next_src_prefix = f"{block_tag}:PostFFN"
    return next_src_prefix, added


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_bias_parameters(model):
    total_bias_params = 0
    bias_list = []
    for name, param in model.named_parameters():
        if "bias" in name and param.requires_grad:
            n_params = param.numel()
            total_bias_params += n_params
            bias_list.append((name, n_params))
    return total_bias_params, bias_list


def save_edges_to_txt(edges, filename):
    with open(filename, "w") as f:
        for src, dst, weight in edges:
            f.write(f"{src}\t{dst}\t{weight}\n")


def save_edges_to_txt_gz(edges, filename):
    with gzip.open(filename, "wt", compresslevel=6) as f:
        for u, v, w in edges:
            f.write(f"{u}\t{v}\t{w}\n")


def main(model_name: str):

    # Load the model
    model, model_info = load_model(model_name=model_name)

    # Collect edges cumulatively
    edges = []

    # --- Embedding stage ---

    add_token_to_preattn_edges(edges, model.embeddings.word_embeddings.weight)  # [V, H]
    add_position_to_preattn_edges(
        edges, model.embeddings.position_embeddings.weight
    )  # [P, H]
    add_type_to_preattn_edges(
        edges, model.embeddings.token_type_embeddings.weight
    )  # [T, H]
    add_layernorm_gamma_self_edges(
        edges, model.embeddings.LayerNorm.weight, "H0"
    )  # [H]

    # Intermediate check (remove later)
    print("Edges after embeddings:", len(edges))
    H_dim = model_info["hidden_size"]
    V_dim = model_info["vocab_size"]
    P_dim = model_info["max_position_embeddings"]
    T_dim = model_info["type_vocab_size"]
    print(
        "Expected after embeddings:",
        V_dim * H_dim + P_dim * H_dim + T_dim * H_dim + H_dim,
    )

    # --- Block stage ---
    num_blocks = model.config.num_hidden_layers

    src_prefix = "H0"
    total_added = 0
    for k in range(num_blocks):
        src_prefix, added = add_block_edges(edges, model, k, src_prefix)
        total_added += added
        print(f"Block {k}: added {added} edges; cumulative = {len(edges)}")

    # --- Pooler ---
    add_postattn_to_pooler_edges(edges, model, src_prefix=src_prefix, target_prefix="P")
    print(f"Total edges after pooler: {len(edges)}")

    # Parameter check
    total_parameters = count_trainable_params(model)
    total_bias, bias_params = count_bias_parameters(model)
    print(f"Total trainable parameters: {total_parameters}")
    print(f"Total trainable bias parameters: {total_bias}")
    # for name, n in bias_params:
    #    print(f"{name}: {n}")
    print(
        f"Total trainable parameters expected: {total_parameters-total_bias} and we got {len(edges)}"
    )

    # Save edges
    if SAVE_MODE:
        save_edges_to_txt(edges, filename=model_name.replace("/", "_") + "_edges.txt")

        # you might want to gzip for huge nets
        # save_edges_to_txt_gz(edges, filename=model_name.replace("/", "_") + "_edges.txt.gz")


if __name__ == "__main__":

    # try these
    # prajjwal1/bert-tiny (L=2, H=128)
    # prajjwal1/bert-mini (L=4, H=256)
    # prajjwal1/bert-small (L=4, H=512)
    # prajjwal1/bert-medium (L=8, H=512)

    parser = argparse.ArgumentParser(description="Run script with a given model name.")
    parser.add_argument("model_name", type=str, help="The name of the model to use.")

    args = parser.parse_args()
    main(args.model_name)
