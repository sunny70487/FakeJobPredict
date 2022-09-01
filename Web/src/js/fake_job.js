
$(function(){
    $("#data").ready(function(){
        $.ajax({
            type: "post",
            url: "http://localhost:5000/data",
            //url: userHost + ":" + userPort + "/login",
            dataType: "json",
            contentType: "application/json",
            success: function(response){
                var aq = "<tr>"+
                        "<th style = 'border: 1px solid black!important;'></th>"+
                        "<th style = 'border: 1px solid black!important;'>title</th>"+
                        "<th style = 'border: 1px solid black!important;'>locaiton</th>"+
                        "<th style = 'border: 1px solid black!important;'>department</th>"+
                        "<th style = 'border: 1px solid black!important;'>salary_range</th>"+
                        "<th style = 'border: 1px solid black!important;'>predict_button</th>"+
                        "</tr>"
                var av = ''
                for (let i = 0; i < response.length; i++) {
                    s = "<tr>"+
                    "<td style = 'border: 1px solid black!important;'>"+(i+1)+"</td>"+
                    "<td style = 'border: 1px solid black!important;'>"+response[i].title+"</td>"+
                    "<td style = 'border: 1px solid black!important;'>"+response[i].location+"</td>"+
                    "<td style = 'border: 1px solid black!important;'>"+response[i].department+"</td>"+
                    "<td style = 'border: 1px solid black!important;'>"+response[i].salary_range+"</td>"+
                    "<td style = 'border: 1px solid black!important;'>"+
                    "<button type='submit' class='btn btn-outline-dark' value ="+response[i].index+">"+"Predict"+
                    "</button>"+
                    "</td>"+
                    "</tr>"
                    av += s;
                e = aq + av
                  }
                $("#data").append(e);
            }
        });

    })
})
$(function(){
    $(document).on("click","#data .btn",function(){
        // 接收前端按鈕傳回的值
        var data =$(this).val();
        console.log(data);
        $.ajax({
            type:'post',
            url: 'http://localhost:5000/execute',
            dataType:'json',
            contentType:'application/json',
            data:JSON.stringify({
                random_num: data
            }),
            success: function(response){
                if(response.predictResult == "1"){
                    Swal.fire({
                        title: "預測成功"+" "+"編號為："+response.index,
                        text: '太好了！！今晚加菜',
                        imageUrl:"assets/img/success.jpg",
                        imageWidth: 400,
                        imageHeight: 200,
                        confirmButtonText: 'OK'
                    })
                //預測失敗
                }else if(response.predictResult == "0"){
                    Swal.fire({
                        title: '預測失敗!'+" "+"編號為："+response.index,
                        text: '革命尚未成功',
                        imageUrl:'assets/img/fail.png',
                        imageWidth: 400,
                        imageHeight: 200,
                        confirmButtonText: 'OK'
                      })
                }
            }


        });

    })
})