from flask import Flask, render_template, request
import os
from extensions import mongo
from flask.logging import default_handler
import ml_new as ml
import learning as ln

app = Flask(__name__)
app.config['MONGO_URI']='mongodb://admin:admin@localhost:4090/admin'
app.config['MONGO_DBNAME']='flask'
mongo.init_app(app)
app.logger.removeHandler(default_handler)
UPLOAD_FOLDER = 'static/files'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# mongo.db.admin.find_one_or_404({'taskId':taskName})


@app.route('/')
def index():
    # global data
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def home():
    global data
    taskName = request.form.get('taskId')
    type = request.form.get('method')
    files = request.files['file']
    app.logger.info('task name is this :{} files :{} type :{}',
                    taskName, files, type)

    if (files.filename != ''):
        if files.filename.endswith('.csv'):    # check if the file is csv
            files.filename = taskName+'.csv'
            file_path = os.path.join(
                app.config['UPLOAD_FOLDER'], files.filename)
            app.logger.debug("file path ", file_path)
            files.save(file_path)
            app.logger.info("success")
    else:
        app.logger.warning("file not uploaded")
        return "Please upload a csv file"
    data = ml.pd.read_csv(file_path)
    app.logger.info("data is :{}", data)
    
    # try:
    #     filenameMongo = mongo.db.flask.find_one_or_404({'taskId':taskName})
    #     if(filenameMongo!=None):
    #         return "Task already exists"
    #     else:
    #         mongo.save_file(files.filename, files,base='fs', content_type='text/csv')
    #         mongoOutput=mongo.db.admin.find_one_or_404({'taskId':taskName})
    #         app.logger.info("mongo output is :{}", mongoOutput)
    # except Exception as e:
    #     # os.remove(file_path)
    #     app.logger.info("this is error -->",e)
    
    return render_template('index.html', df_colum=data.columns)



@app.route('/outputcol', methods=['POST'])
def forOutputCol():
    global df
    app.logger.info('data is here :', data)
    outputColum = request.form.get('output_col')
    columnToBeDropped = request.form.getlist('list')
    if columnToBeDropped != None:
        ml.remove_columns.extend(columnToBeDropped)
    if outputColum != None:
        ml.output_column = outputColum
    app.logger.info('index of output column is :{}', ml.output_column)

    df = ml.data_conversion_and_data_transformation(data)
    df = ml.null_value_count(df)
    printData=df.head(15)
    print(printData)
    return render_template('home.html', tables=[printData.to_html(classes='data',index=False)])


@app.route('/learn', methods=['POST'])
def learning():
    ln.output_column=ml.output_column
    ln.machine_learning_model(df)
    graph=[ln.graphA,ln.graphB,ln.graphC,ln.graphD,ln.graphE]
    output=zip(graph,ln.accuracy)
    return render_template('learn.html',output=output)

@app.route('/test',methods=['get','post'])
def test():
    return render_template('test.html')

@app.route('/home',methods=['get','post'])
def default():
    return render_template('index.html')


if(__name__ == '__main__'):
    app.run(debug=True)
