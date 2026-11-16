from flask import Flask, render_template, request

from recommendation import recommend_papers

app = Flask(__name__)


@app.route("/")
def home():

    return render_template("index.html")


@app.route("/search", methods=["POST"])
def search():

    query = request.form["query"]

    papers = recommend_papers(query)

    return render_template(
        "results.html",
        query=query,
        papers=papers
    )


if __name__ == "__main__":
    app.run(debug=True)