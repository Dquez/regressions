const fs = require('fs');
const _ = require('lodash');

function extractColumns(data, columnNames) {
    const headers = _.first(data);
    const indexes = _.map(columnNames, column => headers.indexOf(column))
    const extracted = _.map(data, row => _.pullAt(row, indexes));
    return extracted;
}
function loadCSV(filename, { dataColumns = [], labelColumns = [], converters = {}}){
    let data = fs.readFileSync(filename, {encoding: 'utf-8'})
    // split on each new line of the csv and make an array of arrays for each element
    data = data.split('\n').map(row => row.split(','))
    // drop trailing empty values in each row
    data.map(row => _.dropRightWhile(row, val => val === ''))
    const headers = _.first(data);

    data = data.map((row,index)=> {
        // row is a header row
        if(index === 0) return row;
        return row.map((element, index) => {
            if(converters[headers[index]]){
                const converted = converters[headers[index]](element);
                return _.isNaN(converted) ? element : converted;
            }
            const result = parseFloat(element);
            return _.isNaN(result) ? element : result;
        })
    })
    let labels = extractColumns(data, labelColumns)
    data = extractColumns(data, dataColumns)
    data.shift();
    labels.shift();
    
    console.log(data);
}   
loadCSV('data.csv', {
    dataColumns: ['height','value'],
    labelColumns: ['passed'], 
    converters : {
        passed: val => val === 'TRUE'
    }
});