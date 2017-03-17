import { Component, Input, Output, EventEmitter, DoCheck } from '@angular/core';

@Component({
    selector: 'pair-list-display',
    template: `
    <div>
        {{name}}:&nbsp;{{_description}}
        <table>
            <tr>
                <td>Xs</td>
                <td *ngFor="let pair of _value; let i=index;">
                    <ndv-edit
                        [title]="'val'"
                        [placeholder]="'' + pair[0]"
                        [permission]="true"
                        (onSave)="update(0, i, $event);">
                    </ndv-edit>
                </td>
            </tr>
            <tr>
                <td>Ys</td>
                <td *ngFor="let pair of _value; let i=index;">
                    <ndv-edit
                        [title]="'val'"
                        [placeholder]="'' + pair[1]"
                        [permission]="true"
                        (onSave)="update(1, i, $event);">
                    </ndv-edit>
                </td>
            </tr>
        </table>
        <button (click)='addPair()'>Add Pair</button>
    </div>
    `
})
export class PairListDisplayComponent {
    @Input() name: string = null;

    @Input()
    set value(value: any) {
        let val = value._value;
        this._description = value.description.value || null;
        this._value = val ? val.pairs : [[0, 0]];
    }

    @Output() publish = new EventEmitter();

    private _value: Array<Array<number>>;
    private _class = ['sifra.structures', 'XYPairs'];
    private _description: string = null;

    addPair() {
        this._value.push([
            this._value[this._value.length-1][0],
            this._value[this._value.length-1][1]
        ]);
        this.doPublish();
    }

    update(row, col, $event) {
        this._value[col][row] = Number($event.val);
        this.doPublish();
    }

    doPublish() {
        this.publish.emit({
            name: this.name,
            value: {class: this._class, pairs: this._value}
        });
    }
}

