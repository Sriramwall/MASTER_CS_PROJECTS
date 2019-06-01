var width = 750,
    height = 500;

var fill = d3.scale.category20();

d3.csv('../wordcloudcsv/cc_full.csv', function (data) {
    var leaders = [];
    data.forEach(function(row){
        if (row.G > 0) leaders.push({text: row.Name, size: Number(row.G)});
    });

    var leaders = leaders.sort(function(a,b){
        return (a.size < b.size)? 1:(a.size == b.size)? 0:-1
    }).slice(0,50);

    var leaderScale = d3.scale.linear()
        .range([20,100])
        .domain([d3.min(leaders,function(d) { return d.size; }),
                 d3.max(leaders,function(d) { return d.size; })
               ]);

    d3.layout.cloud().size([width, height])
        .words(leaders)
        .padding(0)
//      .rotate(function() { return ~~(Math.random() * 2) * 90; })
        .font("Impact")
        .fontSize(function(d) { return leaderScale(d.size); })
        .on("end", drawCloud1)
        .start();
});


function drawCloud1(words) {
    d3.select("#word-cloud1").append("svg")
    .attr("width", width)
    .attr("height", height)
    .append("g")
    .attr("transform", "translate("+(width / 2)+","+(height / 2)+")")
    .selectAll("text")
    .data(words)
    .enter().append("text")
    .style("font-size", function(d) { return d.size + "px"; })
    .style("font-family", "Impact")
    .style("fill", function(d, i) { return fill(i); })
    .attr("text-anchor", "middle")
    .attr("transform", function(d) {
        return "translate(" + [d.x, d.y] + ")rotate(" + d.rotate + ")";
    })
    .text(function(d) { return d.text; });
}

d3.csv('../wordcloudcsv/cc_partial.csv', function (data) {
    var leaders = [];
    data.forEach(function(row){
        if (row.G > 0) leaders.push({text: row.Name, size: Number(row.G)});
    });

    var leaders = leaders.sort(function(a,b){
        return (a.size < b.size)? 1:(a.size == b.size)? 0:-1
    }).slice(0,50);

    var leaderScale = d3.scale.linear()
        .range([20,100])
        .domain([d3.min(leaders,function(d) { return d.size; }),
                 d3.max(leaders,function(d) { return d.size; })
               ]);

    d3.layout.cloud().size([width, height])
        .words(leaders)
        .padding(0)
//      .rotate(function() { return ~~(Math.random() * 2) * 90; })
        .font("Impact")
        .fontSize(function(d) { return leaderScale(d.size); })
        .on("end", drawCloud2)
        .start();
});


function drawCloud2(words) {
    d3.select("#word-cloud2").append("svg")
    .attr("width", width)
    .attr("height", height)
    .append("g")
    .attr("transform", "translate("+(width / 2)+","+(height / 2)+")")
    .selectAll("text")
    .data(words)
    .enter().append("text")
    .style("font-size", function(d) { return d.size + "px"; })
    .style("font-family", "Impact")
    .style("fill", function(d, i) { return fill(i); })
    .attr("text-anchor", "middle")
    .attr("transform", function(d) {
        return "translate(" + [d.x, d.y] + ")rotate(" + d.rotate + ")";
    })
    .text(function(d) { return d.text; });
}
