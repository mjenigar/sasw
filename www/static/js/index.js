INPUT = {
    "input-title": undefined,
    "input-src": undefined,
    "input-date": undefined,
    "input-content": undefined
}

$(document).ready(function(){
    
    $("body").on("click", "#analyze-btn", function(){
        AnalyzeData();
    });

    $("#input-date").datepicker();

});

function GetFormData(){
    let form_flag = true;

    for(key in INPUT){
        let val = $("#"+key).val();
        if((key == "input-title" && val == "") || (key == "input-content" && val == "")){
            $("#"+key).addClass("mandatory-err");
            form_flag = false;
        } else $("#"+key).removeClass("mandatory-err");

        INPUT[key] = $("#"+key).val()
    }

    if(!form_flag) alert("Please fill every mandatory fields!");
    else ClearForm();

    return form_flag;
}

function ClearForm(){
    for(key in INPUT){
        $("#"+key).val("")
    }
}

function AnalyzeData(){
    if(GetFormData()){
        $.ajax({
            url: "/analyze",
            data: {inp: JSON.stringify(INPUT)},
            type: 'POST',
            success: function(data){
                // data = JSON.parse(response);
                console.log(data);
            },
            error: function(error){
                console.log("Something went wrong!");
            }
        });
    } else {
        console.log("NOK");
    } 
}