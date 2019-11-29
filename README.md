# Text-mining_based_legal_assistant_app
 
 
2019/11/29 update

This is a legal assistant, which use text mining and machine learning to predict the possible result of the new case. The user can describe the problems they meet in the life, and the app will give the possible result of appealing, as well as related laws and cases for lawyers. We use crawler to get open data from websites, use TF-IDF to extract the features and key words of former judgement. 

You can see the introduction in the 宣传 folder, and document in the 文件 folder, other related code files are all in the final folder.
The application and server are missing. Only models for prediction still exist.
Just replace the line 131 in the prediction.py and you can see the predicted result of the case you just input.
 
 
The original version fo README
 
这是一个假装很厉害其实很简易的系统

简单来讲就是通过分析原有的案件来预判新案件~

这个readme其实是给要写readme的人看的readme



	→所有代码都在final文件夹

	→说明在文件和宣传

这么做可能可以跑起来？谁知道呢↓

	0. 看看自己有没有Java8、Python环境
	
	1. 安装Apache
	
	2. 编译一下我想毕业server（WXBY_Server)记得要检查一下资料库？？
	
	3. 编译一下我真的想毕业app（GraduationProject里的wozhendexiangbiye）把server的地址改一下~
	
	4. 祈祷没有bug
	
资料库：MongoDB（也不知道还有没有活着）
