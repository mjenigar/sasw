CREATE TABLE articles
(
    ID INT AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(250) NOT NULL,
    content TEXT NOT NULL,
    source varchar(50),
    published DATE,
    analyzed DATE,
    model1 INT NOT NULL,
    model2 INT NOT NULL,
    model3 INT NOT NULL
);

USE sasw;

INSERT INTO articles (title, content, source, published, analyzed model1, model2, model3)
VALUES ('TEST', 'TEST', "reddit",  1, 1, 0);

INSERT INTO articles (title, content, source, model1, model2, model3)
VALUES ('TEST2', 'TEST2', "reddit",  0, 1, 0);