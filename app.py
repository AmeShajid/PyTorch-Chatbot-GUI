from tkinter import *
from chat import get_response, bot_name

# Import everything from Tkinter to build the GUI.
# Import get_response and bot_name functions from the chat module to use in the chat.

BG_GRAY = "#ABB2B9"
BG_COLOR = "#17202A"
TEXT_COLOR = "#EAECEE"

# Define some colors to use in the GUI for a consistent look.
# BG_GRAY: Light gray for certain elements.
# BG_COLOR: Dark background color for the chat window.
# TEXT_COLOR: Light color for text.

FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"

# Define fonts to use for text in the chat window.
# FONT: Regular text font.
# FONT_BOLD: Bold font for headings and buttons.

class ChatApplication:
    
    def __init__(self):
        self.window = Tk()
        self._setup_main_window()
        # Initialize the main chat window and set up its layout.
        
    def run(self):
        self.window.mainloop()
        # Start the Tkinter event loop, which keeps the window open and responsive.
        
    def _setup_main_window(self):
        self.window.title("Chat")
        # Set the title of the window to "Chat".
        
        self.window.resizable(width=False, height=False)
        # Prevent the user from resizing the window. It will stay at a fixed size.
        
        self.window.configure(width=470, height=550, bg=BG_COLOR)
        # Set the size of the window to 470x550 pixels and apply the background color.
        
        # head label
        head_label = Label(self.window, bg=BG_COLOR, fg=TEXT_COLOR,
                           text="Welcome", font=FONT_BOLD, pady=10)
        head_label.place(relwidth=1)
        # Create a label at the top of the window with the text "Welcome".
        # The label spans the full width of the window and has a padding of 10 pixels.
        
        # tiny divider
        line = Label(self.window, width=450, bg=BG_GRAY)
        line.place(relwidth=1, rely=0.07, relheight=0.012)
        # Create a small line to separate the header from the rest of the window.
        # It is placed just below the header, spanning the full width of the window.
        
        # text widget
        self.text_widget = Text(self.window, width=20, height=2, bg=BG_COLOR, fg=TEXT_COLOR,
                                font=FONT, padx=5, pady=5)
        self.text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
        # Create a text area where messages will be displayed.
        # It fills most of the window's height and width.
        
        self.text_widget.configure(cursor="arrow", state=DISABLED)
        # Set the cursor to an arrow when hovering over the text area.
        # Set the text area to be read-only by default (DISABLED).
        
        # scroll bar
        scrollbar = Scrollbar(self.text_widget)
        scrollbar.place(relheight=1, relx=0.974)
        # Add a scrollbar to the text area to navigate through long messages.
        # Place it on the right side of the text area.
        
        scrollbar.configure(command=self.text_widget.yview)
        # Link the scrollbar to the vertical view of the text area so it moves when scrolling.
        
        # bottom label
        bottom_label = Label(self.window, bg=BG_GRAY, height=80)
        bottom_label.place(relwidth=1, rely=0.825)
        # Create a label at the bottom of the window to hold the message entry box and send button.
        # The label spans the full width of the window and has a height of 80 pixels.
        
        # message entry box
        self.msg_entry = Entry(bottom_label, bg="#2C3E50", fg=TEXT_COLOR, font=FONT)
        self.msg_entry.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011)
        # Create a text box where the user types their message.
        # It is placed at the bottom left of the bottom label.
        
        self.msg_entry.focus()
        # Automatically focus on the message entry box when the window opens.
        
        self.msg_entry.bind("<Return>", self._on_enter_pressed)
        # Bind the "Enter" key press event to the _on_enter_pressed method, which will handle sending the message.
        
        # send button
        send_button = Button(bottom_label, text="Send", font=FONT_BOLD, width=20, bg=BG_GRAY,
                             command=lambda: self._on_enter_pressed(None))
        send_button.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)
        # Create a "Send" button next to the message entry box.
        # Clicking the button or pressing "Enter" will trigger the _on_enter_pressed method.
        
    def _on_enter_pressed(self, event):
        msg = self.msg_entry.get()
        # Get the message text from the entry box.
        
        self._insert_message(msg, "You")
        # Insert the user's message into the text area and display it with the sender's label "You".
        
    def _insert_message(self, msg, sender):
        if not msg:
            return
        # If the message is empty, do nothing.
        
        self.msg_entry.delete(0, END)
        # Clear the message entry box after sending the message.
        
        msg1 = f"{sender}: {msg}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg1)
        self.text_widget.configure(state=DISABLED)
        # Format and insert the user's message into the text area.
        # Set the text area to be editable (NORMAL) temporarily to insert the message, then set it back to read-only (DISABLED).
        
        msg2 = f"{bot_name}: {get_response(msg)}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg2)
        self.text_widget.configure(state=DISABLED)
        # Get the bot's response using the get_response function and format it.
        # Insert the bot's response into the text area in the same way as the user's message.
        
        self.text_widget.see(END)
        # Scroll the text area to show the most recent message.
        
if __name__ == "__main__":
    app = ChatApplication()
    app.run()
    # Create an instance of ChatApplication and run it if this script is executed directly.






