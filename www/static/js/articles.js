let MAX_ARTICLE_ON_PAGE = 6
let CURRENT_PAGE = 1
let NO_PAGE = 0

$(document).ready(function(){
    LoadArticles(null);
    Search();
});

function LoadArticles(search){
    console.log("Loading articles...")
    $.ajax({
        url: "/get_records",
        data: {search: search},
        type: 'POST',
        success: function(data){
            data = JSON.parse(data);
            console.log(data);
            console.log("data len: " + data.length);
            let HTML = '';
            NO_PAGE = Math.ceil(data.length/MAX_ARTICLE_ON_PAGE)
            let on_page = 0;
            let page = 1;
            let html = '<div class="row" id="page-'+page++ +'"><div class="row w-100">';
            for(let i = 0; i < data.length; i++){
                ++on_page;
                if(on_page > MAX_ARTICLE_ON_PAGE){
                    let class__ = (page != CURRENT_PAGE) ? "hidden": "";
                    html += '</div></div><div class="row '+class__+'" id="page-'+ page++ +'"><div class="row w-100">';
                    on_page = 1;
                }
                
                let class_ = (i == (data.length - 1) && i % 2 == 0) ? "col": "col-sm-6";
                let side = (i % 2 == 0) ? "float-left" : "float-right";
                html += ''+
                    '<div id="record-'+data[i][0]+'" class="'+class_+' p-3 '+side+'">' +
                    '   <div class="card bg-dark shadow-lg rounded" style="height: 600px !important; width: 650px !important; text-align: start !important;">' +
                    '       <div class="card-header mb-2">'+
                    '           <div class="row"><div class="col"><h2 class="card-title">'+ data[i][1] +'</h2></div></div>' +
                    '           <div class="row" style="font-style: italic;"><div class="col text-muted start"> Published: '+FormatDate(data[i][4])+'</div><div class="col text-muted center"> Analyzed: '+FormatDate(data[i][5])+'</div><div class="col text-muted end""> Source: '+data[i][3]+'</div></div>'+
                    '       </div>'+
                    '       <div class="card-body" style="overflow-x: hidden; overflow-y: auto;">' +
                    '           <div class="row"><div class="col"><p class="card-text" style="text-align: justify;">'+data[i][2]+'</p></div></div>' +
                    '       </div>' +
                    '       <div class="card-footer center">'+
                    '           <div class="row"><div class="col" data-toggle="tooltip" title="Random Forest Classifier">'+ GetIcon(data[i][6]) +'</div><div class="col" data-toggle="tooltip" title="Naive Bayes Classifier">'+ GetIcon(data[i][7]) +'</div><div class="col" data-toggle="tooltip" title="Recurrent Neural Net">'+ GetIcon(data[i][8]) +'</div></div>' +
                    '       </div>'+
                    '   </div>'+
                    '</div>';
                
                if(on_page % 2 == 0 && on_page < MAX_ARTICLE_ON_PAGE) html += '</div><div class="row w-100">';
            }
            $("#article-field").empty().append(html);
            $('[data-toggle="tooltip"]').tooltip();

            RenderPagination();
        },
        error: function(error){
            console.log("Something went wrong!");
        }
    });
}

function FormatDate(date){
    if(date == null) return "";

    split = date.split("-");
    return split[2]+"."+split[1]+"."+split[0];
}

function GetIcon(prediction){
    return (prediction == 1) ? '<i class="fas fa-thumbs-up text-success"></i>' : '<i class="fas fa-thumbs-down text-danger"></i>';
}

function RenderPagination(){
    let next_disabled = (CURRENT_PAGE == NO_PAGE) ? "disabled" : "";
    let prev_disabled = (CURRENT_PAGE == 1) ? "disabled" : "";

    let html = '<ul class="pagination justify-content-center">' +
                    '<li class="page-item '+prev_disabled+'">' +
                        '<a class="page-link bg-secondary" onclick="PrevPage()" tabindex="-1"><i class="fas fa-chevron-left text-dark"></i></a>' +
                '   </li>';
        for(var i = 1; i <= NO_PAGE; i++){
            active = (i == CURRENT_PAGE) ? "active-page" : "bg-dark";
            html += '<li id="page-btn-'+i+'" class="page-item" onclick="Go2Page(this.id)"><a id="page-anchor-'+i+'" class="page-link '+active+' text-secondary">'+i+'</a></li>';
        }        
        html += '   <li class="page-item '+next_disabled+'">' +
                        '<a class="page-link bg-secondary" onclick="NextPage()"><i class="fas fa-chevron-right text-dark"></i></a>' +
                '   </li>'+
                '</ul>';
    
    $("#pagination-field").empty().append(html);
}

function NextPage(){
    if((CURRENT_PAGE+1) > NO_PAGE) return;
    
    $("#page-"+CURRENT_PAGE++).addClass("hidden");
    $("#page-"+CURRENT_PAGE).removeClass("hidden");

    RenderPagination();
}

function PrevPage(){
    if((CURRENT_PAGE-1) <= 0) return;
    
    $("#page-"+CURRENT_PAGE--).addClass("hidden");
    $("#page-"+CURRENT_PAGE).removeClass("hidden");

    RenderPagination();
}

function Go2Page(id){
    id = id.split("-")[2]

    $("#page-"+CURRENT_PAGE).addClass("hidden");
    CURRENT_PAGE = Number(id);
    $("#page-"+CURRENT_PAGE).removeClass("hidden");
    RenderPagination();
}

function Search()
{
    let inputStr = "";
    let typingTimer;                
    $("#search-input").on("change keyup paste", function(){
        if(inputStr == $('#search-input').val()) return;
        else inputStr = $('#search-input').val();

        if (inputStr) {
            typingTimer = setTimeout(DoneTyping, 1000);
        } else if(inputStr == "") return
    });
}

function DoneTyping()
{
    let inputStr = $('#search-input').val();
    console.log(inputStr)
    if (inputStr.length <= 0){
        inputStr = null;
        LoadArticles(inputStr);
        return;
    }
    LoadArticles(inputStr);
}
