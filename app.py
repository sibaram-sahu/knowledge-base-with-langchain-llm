import streamlit as st


def main():
    st.set_page_config(page_title="Conversation with Knowledge Base", page_icon=":books:")

    st.header("Conversation with Knowledge Base :books:")
    st.text_input("Write a query about your documents.")

    with st.sidebar:
        st.subheader('Knowledge Base')
        st.file_uploader("Upload your Knowledge Base(pdfs)")
        st.button("Upload")

if __name__ == "__main__":
    main()