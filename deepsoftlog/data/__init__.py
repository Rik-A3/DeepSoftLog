from deepsoftlog.algebraic_prover.terms.expression import Constant, Expr

from .query import Query
from ..logic.soft_term import SoftTerm, TensorTerm


def nl2tsv(src_path, dst_path):
    with open(src_path, "r") as f_in:
        with open(dst_path, "w") as f_out:
            for line in f_in:
                line = line.strip()
                arr = line.replace("(", " ").replace(")", " ").replace(",", " ").replace(".","").split(" ")
                arr[0], arr[1] = arr[1], arr[0]
                f_out.write("\t".join(arr) + "\n")

def load_tsv_file(filename: str):
    with open(filename, "r") as f:
        return [line.strip().split("\t") for line in f.readlines()]

def load_txt_file(filename: str):
    with open(filename, "r") as f:
        return [line.strip().split("\t") for line in f.readlines()]

def load_lines(filename: str):
    with open(filename, "r") as f:
        return [line for line in f.readlines()]

def data_to_prolog(rows, name="r", **kwargs):
    for row in rows:
        args = [Constant(a) for a in row]
        args = [args[1], args[0], args[2]]
        args = [SoftTerm(a) for a in args]
        yield Query(Expr(name, *args), **kwargs)

def to_prolog_image(img):
    return SoftTerm(Expr("lenet5", TensorTerm(img)))

def to_prolog_text(text):
    return SoftTerm(Expr("roberta", TensorTerm(text)))
