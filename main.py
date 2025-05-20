import tkinter as tk
from tkinter import messagebox, simpledialog
import numpy as np
from tkinter import font as tkfont

# ---------- الگوریتم استراسن با padding ----------
def add(A, B):
    return A + B

def subtract(A, B):
    return A - B

def pad_to_power_of_two(A, B):
    r1, c1 = A.shape
    r2, c2 = B.shape
    n = 1 << max(r1, c1, r2, c2).bit_length()
    A_pad = np.zeros((n, n))
    B_pad = np.zeros((n, n))
    A_pad[:r1, :c1] = A
    B_pad[:r2, :c2] = B
    return A_pad, B_pad, r1, c2

def strassen(A, B, threshold=64):
    n = A.shape[0]
    if n <= threshold:
        return A @ B
    mid = n // 2
    A11, A12 = A[:mid, :mid], A[:mid, mid:]
    A21, A22 = A[mid:, :mid], A[mid:, mid:]
    B11, B12 = B[:mid, :mid], B[:mid, mid:]
    B21, B22 = B[mid:, :mid], B[mid:, mid:]
    M1 = strassen(add(A11, A22), add(B11, B22), threshold)
    M2 = strassen(add(A21, A22), B11, threshold)
    M3 = strassen(A11, subtract(B12, B22), threshold)
    M4 = strassen(A22, subtract(B21, B11), threshold)
    M5 = strassen(add(A11, A12), B22, threshold)
    M6 = strassen(subtract(A21, A11), add(B11, B12), threshold)
    M7 = strassen(subtract(A12, A22), add(B21, B22), threshold)
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    top = np.hstack((C11, C12))
    bottom = np.hstack((C21, C22))
    return np.vstack((top, bottom))

# ---------- ماتریس مقدماتی ----------
def make_elementary_swap(n, i, j):
    E = np.eye(n)
    E[[i, j]] = E[[j, i]]
    return E

def make_elementary_scale(n, i, scalar):
    E = np.eye(n)
    E[i, i] = scalar
    return E

def make_elementary_add(n, src, target, scalar):
    E = np.eye(n)
    E[target, src] = scalar
    return E

def gauss_jordan_with_elementary(A, B):
    A = A.astype(float)
    B = B.astype(float)
    n = A.shape[0]
    augmented = np.hstack((A, B))
    E_total = np.eye(n)
    for i in range(n):
        if augmented[i, i] == 0:
            for j in range(i + 1, n):
                if augmented[j, i] != 0:
                    E = make_elementary_swap(n, i, j)
                    augmented = E @ augmented
                    E_total = E @ E_total
                    break
        pivot = augmented[i, i]
        if pivot != 1 and pivot != 0:
            E = make_elementary_scale(n, i, 1 / pivot)
            augmented = E @ augmented
            E_total = E @ E_total
        for j in range(n):
            if j != i and augmented[j, i] != 0:
                factor = -augmented[j, i]
                E = make_elementary_add(n, i, j, factor)
                augmented = E @ augmented
                E_total = E @ E_total
    rank = np.linalg.matrix_rank(A)
    X = augmented[:, n:]
    A_inv = E_total @ np.eye(n)
    return X, A_inv, rank

def calculate_determinant(A):
    return round(np.linalg.det(A), 4)

def calculate_inverse(A):
    return np.linalg.inv(A)

# ---------- رابط گرافیکی ----------
class MatrixApp:
    def __init__(self, root):
        self.root = root
        root.title("ماشین حساب ماتریسی")
        root.geometry("800x600")
        root.configure(bg="#222")

        self.font = tkfont.Font(family="B Titr", size=14)

        self.label = tk.Label(root, text="یک عملیات انتخاب کنید:", font=self.font, bg="#222", fg="white")
        self.label.pack(pady=10)

        self.frame = tk.Frame(root, bg="#222")
        self.frame.pack()

        self.text = tk.Text(root, height=12, font=self.font, bg="#111", fg="white")
        self.text.pack(pady=20, fill=tk.BOTH, expand=True)

        buttons = [
            ("حل معادله", self.solve),
            ("محاسبه معکوس ماتریس", self.inverse),
            ("محاسبه دترمینان", self.determinant)
        ]
        for text, cmd in buttons:
            tk.Button(self.frame, text=text, command=cmd, font=self.font, bg="#444", fg="white",
                      activebackground="#666", activeforeground="white", padx=10, pady=5).pack(pady=5, fill=tk.X)

    def get_matrix(self, prompt):
        text = simpledialog.askstring("ورودی ماتریس", prompt)
        if text is None:
            return None
        try:
            rows = text.strip().split(';')
            matrix = [list(map(float, row.split())) for row in rows]
            return np.array(matrix)
        except:
            messagebox.showerror("خطا", "فرمت ورودی نادرست است.")
            return None

    def solve(self):
        A = self.get_matrix("ماتریس A را وارد کنید (برای مثال: 2 1; 1 3)")
        B = self.get_matrix("ماتریس B را وارد کنید (برای مثال: 5; 7)")
        if A is None or B is None:
            return
        try:
            if B.ndim == 1:
                B = B.reshape(-1, 1)
            X, A_inv, rank = gauss_jordan_with_elementary(A, B)
            if rank < A.shape[0]:
                self.text.insert(tk.END, f"\nمعادله بی‌نهایت جواب دارد چون رنک ({rank}) < تعداد سطرها ({A.shape[0]})\n")
            else:
                self.text.insert(tk.END, f"\nحل معادله:\nX = {X}\n")
        except Exception as e:
            messagebox.showerror("خطا", str(e))

    def inverse(self):
        A = self.get_matrix("ماتریس مربعی A را وارد کنید")
        if A is None:
            return
        try:
            inv = calculate_inverse(A)
            self.text.insert(tk.END, f"\nمعکوس ماتریس:\n{inv}\n")
        except Exception as e:
            messagebox.showerror("خطا", str(e))

    def determinant(self):
        A = self.get_matrix("ماتریس مربعی A را وارد کنید")
        if A is None:
            return
        try:
            det = calculate_determinant(A)
            self.text.insert(tk.END, f"\nدترمینان ماتریس: {det}\n")
        except Exception as e:
            messagebox.showerror("خطا", str(e))