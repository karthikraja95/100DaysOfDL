import streamlit as st


def main():
    # Headings for Web Application

    st.title("Semantic Search Engine to Find Similar Complaints")
    st.subheader("Select Model")
    st.text("Right now we have two BERT based Model")
    st.text("1. BASE BERT")
    st.text("2. RoBERTa")

    option = st.selectbox("Model", ("Base BERT", "RoBERTa"))

    # Textbox for the User Query

    st.subheader("Enter the text you would like to find Smiliar Complaints?")
    query = st.text_input("Enter Search Keyword or Sentence")

    st.subheader("How many Complaints do you want you see?")
    number = st.number_input("Enter number")
    st.write("The number entered is", number)

    # Display the Similar Complaints
    st.subheader("The Top Complaints are: ")


if __name__ == "__main__":
    main()
