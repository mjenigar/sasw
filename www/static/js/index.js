INPUT = {
    "input-title": undefined,
    "input-src": undefined,
    "input-date": undefined,
    "input-content": undefined
}

$(document).ready(function(){
    $("body").on("click", "#analyze-btn", function(){
        console.log("analyze");
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
        // $(".wrapper").removeClass("hidden");
        // $("#loader-container").removeClass("hidden");
        $.ajax({
            url: "/analyze",
            data: {input: JSON.stringify(INPUT)},
            type: 'POST',
            success: function(data){
                // data = JSON.parse(data);
                console.log(data);
                // RenderReport(data);
                // $("#main-modal").modal('show');
                
            },
            error: function(error){
                console.log("Something went wrong!");
            }
        });
    } else {
        console.log("NOK");
    } 
}

function RenderReport(data){
    let HTML = '';
    let idx = 0;
    let models = ['Random Forest', 'Naive Bayes', 'Neural Net'];
    
    HTML += '<div class="col">'
    for(key in data){
        HTML += '' + 
            '<div class="row report-row">' +
            '   <div class="col-sm-10">'+models[idx++]+'</div>'+
            '   <div class="col-sm-1">'+GetIcon(data[key])+'</div>'+
            '</div>';
    }
    HTML += "</div>";
    
    $("#modal-body").empty().append(HTML);
}

function GetIcon(prediction){
    return Number(prediction) == 1 ? '<i class="fas fa-thumbs-up text-success"></i>' : '<i class="fas fa-thumbs-down text-danger"></i>'
}

function GetColor(prediction){
    return Number(prediction) == 1 ? "text-success" : "text-danger";
}