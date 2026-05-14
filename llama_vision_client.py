import base64
import json
import mimetypes
import os
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None
    ImageTk = None


DEFAULT_SERVER = "127.0.0.1:8080"
MAX_PREVIEW_SIZE = (560, 360)


class LlamaVisionClient(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("llama.cpp Vision Client")
        self.geometry("760x720")
        self.minsize(640, 560)

        self.image_path = None
        self.preview_image = None

        self.server_var = tk.StringVar(value=DEFAULT_SERVER)
        self.use_image_url_var = tk.BooleanVar(value=False)

        self._build_ui()

    def _build_ui(self):
        top_bar = tk.Frame(self, padx=12, pady=10)
        top_bar.pack(fill=tk.X)

        tk.Button(top_bar, text="Load Image", command=self.load_image).pack(side=tk.LEFT)
        tk.Button(top_bar, text="Run", command=self.run_request).pack(side=tk.LEFT, padx=(8, 0))

        tk.Checkbutton(
            top_bar,
            text="Use image URL",
            variable=self.use_image_url_var,
        ).pack(side=tk.LEFT, padx=(16, 0))

        tk.Label(top_bar, text="Server").pack(side=tk.LEFT, padx=(16, 4))
        tk.Entry(top_bar, textvariable=self.server_var, width=24).pack(side=tk.LEFT)

        self.image_label = tk.Label(
            self,
            text="No image loaded",
            anchor=tk.CENTER,
            relief=tk.SUNKEN,
            bg="#f4f4f4",
            width=80,
            height=20,
        )
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 10))

        self.result_text = scrolledtext.ScrolledText(self, height=14, wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=False, padx=12, pady=(0, 12))

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.webp *.bmp *.gif"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        self.image_path = path
        self._show_preview(path)
        self._set_result(f"Loaded image:\n{path}\n")

    def _show_preview(self, path):
        if Image and ImageTk:
            image = Image.open(path)
            image.thumbnail(MAX_PREVIEW_SIZE)
            self.preview_image = ImageTk.PhotoImage(image)
            self.image_label.configure(image=self.preview_image, text="", bg="#ffffff")
            return

        try:
            self.preview_image = tk.PhotoImage(file=path)
            self.image_label.configure(image=self.preview_image, text="", bg="#ffffff")
        except tk.TclError:
            self.preview_image = None
            self.image_label.configure(
                image="",
                text=(
                    f"Loaded: {os.path.basename(path)}\n\n"
                    "Install Pillow to preview this image type."
                ),
                bg="#f4f4f4",
            )

    def run_request(self):
        if not self.image_path:
            messagebox.showwarning("Missing image", "Load an image before running the request.")
            return

        server = self.server_var.get().strip()
        if not server:
            messagebox.showwarning("Missing server", "Enter a server IP and port.")
            return

        self._set_result("Sending request...\n")
        self._set_controls_enabled(False)

        worker = threading.Thread(
            target=self._run_request_worker,
            args=(server, self.image_path, self.use_image_url_var.get()),
            daemon=True,
        )
        worker.start()

    def _run_request_worker(self, server, image_path, use_image_url):
        start = time.perf_counter()
        try:
            payload = self._build_payload(server, image_path, use_image_url)
            endpoint = f"http://{server.rstrip('/')}/v1/chat/completions"

            request = Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urlopen(request, timeout=600) as response:
                raw_response = response.read().decode("utf-8", errors="replace")

            elapsed = time.perf_counter() - start
            response_json = json.loads(raw_response)
            text = self._extract_text(response_json)

            mode = "image_url" if use_image_url else "base64 data URL"
            output = (
                f"Completed in {elapsed:.3f} seconds ({elapsed * 1000:.0f} ms)\n"
                f"Endpoint: {endpoint}\n"
                f"Mode: {mode}\n\n"
                f"{text}\n"
            )
        except HTTPError as exc:
            elapsed = time.perf_counter() - start
            error_body = exc.read().decode("utf-8", errors="replace")
            output = (
                f"HTTP error after {elapsed:.3f} seconds ({elapsed * 1000:.0f} ms): "
                f"{exc.code} {exc.reason}\n\n{error_body}\n"
            )
        except (URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError) as exc:
            elapsed = time.perf_counter() - start
            output = f"Request failed after {elapsed:.3f} seconds ({elapsed * 1000:.0f} ms):\n{exc}\n"

        self.after(0, self._finish_request, output)

    def _build_payload(self, server, image_path, use_image_url):
        image_ref = self._image_url(server, image_path) if use_image_url else self._image_data_url(image_path)

        return {
            "model": "llama.cpp",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": ""},
                        {"type": "image_url", "image_url": {"url": image_ref}},
                    ],
                }
            ],
            "stream": False,
        }

    def _image_data_url(self, image_path):
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "application/octet-stream"

        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("ascii")

        return f"data:{mime_type};base64,{encoded}"

    def _image_url(self, server, image_path):
        return f"http://{server.rstrip('/')}/{os.path.basename(image_path)}"

    def _extract_text(self, response_json):
        choices = response_json.get("choices", [])
        if not choices:
            return json.dumps(response_json, indent=2)

        message = choices[0].get("message", {})
        content = message.get("content", "")

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            if parts:
                return "\n".join(parts)

        return json.dumps(response_json, indent=2)

    def _finish_request(self, output):
        self._set_result(output)
        self._set_controls_enabled(True)

    def _set_result(self, text):
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, text)

    def _set_controls_enabled(self, enabled):
        state = tk.NORMAL if enabled else tk.DISABLED
        for child in self.winfo_children():
            self._set_widget_state(child, state)
        self.result_text.configure(state=tk.NORMAL)

    def _set_widget_state(self, widget, state):
        try:
            widget.configure(state=state)
        except tk.TclError:
            pass

        for child in widget.winfo_children():
            self._set_widget_state(child, state)


if __name__ == "__main__":
    app = LlamaVisionClient()
    app.mainloop()
