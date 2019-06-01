//var display = d3.select('input[name="mode2"]:checked').node().value;
var display = d3.select('#dropdown option:checked').text();

var dataDomain;

/* Define SVG elements */
var diameter = 700,
    format = d3.format(",d");
    color = d3.scale.category20c();  // experimenting with color scale later.

var svg = d3.select("#bigram").append("svg")
              .attr("width", diameter)
              .attr("height", diameter)
              .attr("class", "bubble");

var bubble = d3.layout.pack()
              .sort(null) // random sort, if omitted then spirals out inner to outer small to large
              //.sort(function(a,b) {return -(a.value-b.value); }) // smaller on outside...think tail!
              .size([diameter, diameter])
              .padding(1.5);

d3.json("../jsonFiles/twitter_ncaa_ncaa.json", function(data) {
  drawBubbles(data);
});

// Radio buttons
d3.select("#dropdown").on("change", function change() {
  display = d3.select('#dropdown option:checked').text();
  console.log(display);
  if (display) {
    d3.json("../jsonFiles/twitter_ncaa_".concat(display).concat('.json'), function(data) {
      drawBubbles(data);
    });
  }  else { //undefined or total
    d3.json("../jsonFiles/twitter_ncaa_ncaa.json", function(data) {
      drawBubbles(data);
    });
  }
});

function drawBubbles(data) {

    //console.log(data);

    var duration = 0;
    var delay = 0;

  // set data
d3.selectAll(".node").remove();
var node = svg.selectAll("circle")
    .data(bubble.nodes(classes(data))
    .filter(function(d) { return !d.children; }))
  .enter().append("g")
    .attr("class", "node")  // is used?
    .attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; }); // positions the bubbles

  node.append("title") // is used?
    .text(function(d) { return d.className + ": " + format(d.value); });

  // Draw and color circles for nodes
  node.append("circle")
    .attr("r", function(d) { return d.r; })
    .style("fill", function(d) { return color(d.className); }); //color each bubble //return color(d.packageName);
     //.style("fill", function(d) { return colorScale(d.value); });

  node.append("text")
    .attr("dy", ".3em")
    .style("text-anchor", "middle")
    .text(function(d) { return d.className.substring(0, d.r / 3); });

    node.transition()
       .duration(duration)
       .delay(function(d, i) {
      		delay = i * 7;
      		return delay;
    	 })
       .attr("transform", function(d) {
      		return "translate(" + d.x + "," + d.y + ")";
       })
       .attr("r", function(d) { return d.r; })


}

function classes(root) {
  var classes = [];
  function recurse(name, node) {
    if (node.children) node.children.forEach(function(child) {
      recurse(node.word, child);
      //console.log(node.name);
    });
    else {
      //console.log(node.name);
      classes.push({
        packageName: name,
        className: node.word,
        value: node.count
      });
    } // end else
  } // end for each

  recurse(null, root);
  return { children: classes };
}

d3.select(self.frameElement).style("height", diameter + "px");












var width = 750,
    height = 500;

var fill = d3.scale.category20();

d3.csv('../wordcloudcsv/twitter_ncaa.csv', function (data) {
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
        .on("end", drawCloud)
        .start();
});


function drawCloud(words) {
    d3.select("#word-cloud").append("svg")
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
