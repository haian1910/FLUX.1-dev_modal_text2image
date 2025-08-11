# Modal App

This is a simple example project using [Modal](https://modal.com/) for serverless Python execution.

## 📁 Files

- `app_modal.py` – The main application code using Modal.
- `test.py` – A test script to verify the Modal function.

---

## ⚙️ Setup Instructions

### 1. Install Dependencies

```bash
pip install modal
```

### 2. Log In to Modal

If you haven't already:

```bash
modal setup
```

Follow the instructions in the terminal to authenticate and set up your account.

### 3. Set Up Secret

Setup in modal web 


## 📝 Write Your App

In `app_modal.py`, define your Modal function.


---

## 🚀 Deploy the App

Upload and deploy the app to Modal:

```bash
modal deploy app_modal.py
```

---

## 🧪 Run the Test(can skip building app and just run test in order to test sd 1.5 api )

After deploying, run your test script:

```bash
python test.py
```